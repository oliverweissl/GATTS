import torch
import whisper

class AutomaticSpeechRecognitionModel:
    def __init__(self, model_name="tiny", device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = whisper.load_model(model_name, device=device)

    def analyzeAudio(self, wav: torch.Tensor) -> tuple[dict, float]:
        result = self.model.transcribe(audio=wav)

        total_logprob = 0.0
        total_tokens = 0
        for seg in result["segments"]:
            total_logprob += seg["avg_logprob"] * len(seg["tokens"])
            total_tokens += len(seg["tokens"])

        avg_logprob = total_logprob / total_tokens if total_tokens > 0 else float("nan")
        return result, avg_logprob