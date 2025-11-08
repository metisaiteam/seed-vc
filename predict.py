import torch
import torchaudio
import librosa
import yaml
import numpy as np
from pathlib import Path as PathlibPath
import tempfile
import logging
from typing import Optional
from cog import BasePredictor, Path, Input

# Import the necessary modules from seed-vc
from modules.commons import recursive_munch, str2bool
from hf_utils import load_custom_model_from_hf

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and timestamps."""
    COLORS = {
        'INFO': '\033[94m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        record.timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        formatted_message = f"{color}[{record.levelname}] {record.timestamp} - {record.getMessage()}{reset}"
        return formatted_message

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)

class Predictor(BasePredictor):
    def setup(self):
        """Load the Seed-VC model and all necessary components."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model configuration and checkpoints
        logger.info("Loading Seed-VC model...")
        
        # Load DiT checkpoint and config
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
        )
        
        # Load configuration
        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        self.sr = config["preprocess_params"]["sr"]
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        
        # Build model
        from modules.commons import build_model, load_checkpoint
        model = build_model(model_params, stage="DiT")
        
        # Load checkpoint
        model, _, _, _ = load_checkpoint(
            model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        
        # Setup caches for the model
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        self.model = model
        logger.info("Model loaded successfully")
        
        # Load CAMPPlus speaker encoder
        logger.info("Loading speaker encoder...")
        from modules.campplus.DTDNN import CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", 
            "campplus_cn_common.bin", 
            config_filename=None
        )
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.device)
        logger.info("Speaker encoder loaded")
        
        # Load vocoder (BigVGAN)
        logger.info("Loading vocoder...")
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        self.vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.device)
        logger.info("Vocoder loaded")
        
        # Load Whisper for speech tokenization
        logger.info("Loading Whisper model...")
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        self.whisper_model = WhisperModel.from_pretrained(
            whisper_name, 
            torch_dtype=torch.float16
        ).to(self.device)
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        logger.info("Whisper model loaded")
        
        # Setup mel spectrogram function
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr,
            "fmin": config['preprocess_params'].get('fmin', 0),
            "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        from modules.audio import mel_spectrogram
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
        
        logger.info("Setup complete!")

    def extract_semantic_features(self, waves_16k):
        """Extract semantic features using Whisper."""
        ori_inputs = self.whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True
        )
        ori_input_features = self.whisper_model._mask_input_features(
            ori_inputs.input_features, 
            attention_mask=ori_inputs.attention_mask
        ).to(self.device)
        
        with torch.no_grad():
            ori_outputs = self.whisper_model.encoder(
                ori_input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
        return S_ori

    def predict(
        self,
        source_audio: Path = Input(description="Source audio file to convert"),
        reference_audio: Path = Input(description="Reference audio file for target voice"),
        diffusion_steps: int = Input(
            description="Number of diffusion steps (higher = better quality, slower)",
            default=30,
            ge=10,
            le=100
        ),
        length_adjust: float = Input(
            description="Length adjustment factor",
            default=1.0,
            ge=0.5,
            le=2.0
        ),
        inference_cfg_rate: float = Input(
            description="Classifier-free guidance rate",
            default=0.7,
            ge=0.0,
            le=1.0
        ),
    ) -> Path:
        """Run voice conversion from source to reference voice."""
        
        logger.info("Loading audio files...")
        
        # Load and process source audio
        source_audio_np, _ = librosa.load(str(source_audio), sr=self.sr)
        source_audio_tensor = torch.tensor(source_audio_np).unsqueeze(0).float().to(self.device)
        
        # Load and process reference audio (limit to 25 seconds)
        ref_audio_np, _ = librosa.load(str(reference_audio), sr=self.sr)
        ref_audio_np = ref_audio_np[:self.sr * 25]
        ref_audio_tensor = torch.tensor(ref_audio_np).unsqueeze(0).float().to(self.device)
        
        logger.info("Extracting features...")
        
        # Resample to 16kHz for feature extraction
        source_waves_16k = torchaudio.functional.resample(source_audio_tensor, self.sr, 16000)
        ref_waves_16k = torchaudio.functional.resample(ref_audio_tensor, self.sr, 16000)
        
        # Extract semantic features
        S_alt = self.extract_semantic_features(source_waves_16k)
        S_ori = self.extract_semantic_features(ref_waves_16k)
        
        # Extract mel spectrograms
        mel = self.mel_fn(source_audio_tensor)
        mel2 = self.mel_fn(ref_audio_tensor)
        
        # Calculate lengths
        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(self.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(self.device)
        
        # Extract speaker style from reference
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k, 
            num_mel_bins=80, 
            dither=0, 
            sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))
        
        logger.info("Running voice conversion...")
        
        # Length regulation
        with torch.no_grad():
            cond, _, _, _, _ = self.model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=None
            )
            prompt_condition, _, _, _, _ = self.model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
            )
            
            # Concatenate conditions
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            
            # Run flow matching for voice conversion
            vc_target = self.model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1):]
            
            # Generate waveform using vocoder
            vc_wave = self.vocoder(vc_target).squeeze(1)
        
        logger.info("Saving output...")
        
        # Save output
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        torchaudio.save(output_path, vc_wave.cpu(), self.sr)
        
        logger.info("Voice conversion complete!")
        
        return Path(output_path)