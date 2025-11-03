import cog
import torch
from transformers import AutoModel, AutoProcessor
import librosa
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
import os
import tempfile
from typing import Any

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model and processor on startup."""
        model_id = "Plachta/Seed-VC"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor (handles audio tokenization/feature extraction)
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Load model (voice conversion transformer/diffusion)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        source_audio: cog.Path = cog.Path(),  # Input speech audio file
        reference_audio: cog.Path = cog.Path(),  # Reference voice audio file
    ) -> cog.Path:
        """Run voice conversion."""
        if not source_audio or not reference_audio:
            raise ValueError("Both source_audio and reference_audio are required.")

        # Load and preprocess reference audio (extract speaker embedding)
        ref_audio, ref_sr = librosa.load(reference_audio, sr=16000, mono=True)
        ref_audio = librosa.util.normalize(ref_audio)  # Normalize
        ref_inputs = self.processor(ref_audio, sampling_rate=16000, return_tensors="pt")
        ref_inputs = {k: v.to(self.device) for k, v in ref_inputs.items()}
        
        with torch.no_grad():
            ref_features = self.model.get_speaker_embedding(**ref_inputs)  # Extract embedding

        # Load and preprocess source audio
        src_audio, src_sr = librosa.load(source_audio, sr=16000, mono=True)
        src_audio = librosa.util.normalize(src_audio)  # Normalize
        src_inputs = self.processor(src_audio, sampling_rate=16000, return_tensors="pt")
        src_inputs = {k: v.to(self.device) for k, v in src_inputs.items()}

        # Run voice conversion
        with torch.no_grad():
            converted_audio = self.model.generate(
                **src_inputs,
                speaker_embedding=ref_features,  # Apply target voice
                guidance_scale=1.0,  # Default; can expose as input if needed
                max_new_tokens=len(src_audio),  # Match source length
            ).cpu().numpy().squeeze()

        # Post-process: Resample if needed and save as WAV
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        write(output_path, 16000, (converted_audio * 32767).astype(np.int16))  # 16-bit PCM

        return cog.Path(output_path)
