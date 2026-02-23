from faster_whisper import WhisperModel
import os

class TranscriptionEngine:
    def __init__(self, model_size, device="cpu", compute_type="int8", download_root=None):
        self.model = None
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root

    def load_model(self):
        """Loads the Whisper model with error handling."""
        # Check for TensorRT optimization if device is cuda
        # CTranslate2 handles this if the model is converted or if using specific options,
        # but standard faster-whisper usage relies on compute_type.
        print(f"Loading model {self.model_size} on {self.device} ({self.compute_type})...")
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root
            )
            return self.model
        except Exception as e:
            msg = str(e)
            # Automatic Fallback for float16 error on older GPUs
            if "float16" in msg and self.device == "cuda" and self.compute_type == "float16":
                print(f"Float16 failed ({msg}). Retrying with float32...")
                try:
                    self.compute_type = "float32"
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=self.download_root
                    )
                    return self.model
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model with fallback float32: {e2}")

            if "int8" in self.compute_type and self.device == "cuda":
                raise RuntimeError(f"Failed to load model on CUDA with int8. Try float16 or int8_float16. Error: {msg}")
            if "disk space" in msg.lower():
                raise RuntimeError(f"Insufficient disk space to download model. Error: {msg}")
            raise RuntimeError(f"Failed to load/download model '{self.model_size}': {msg}")

    def transcribe(self, file_path, **kwargs):
        """Wraps model.transcribe."""
        if not self.model:
            raise RuntimeError("Model not loaded.")

        # Check for translation request
        # If task is not explicitly set, or if we want to force English
        # We can implement auto-translate logic here if needed, but usually passed via kwargs

        return self.model.transcribe(file_path, **kwargs)

    def close(self):
        """Releases resources."""
        import gc
        if self.model:
            del self.model
            self.model = None
        gc.collect()
        if self.device == "cuda":
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
