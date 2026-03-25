import sys
from typing import Any
import numpy as np
import whisper

_whisper_model = None

def _load_model():
    global _whisper_model, _whisper_pipeline
    if _whisper_model is None:
        _whisper_model = whisper.load_model("tiny", device="cuda")


def whisper_ASR(audio_file: Any) -> str:
    _load_model()
    if isinstance(audio_file, np.ndarray) and audio_file.dtype == np.int16:
        audio_file = audio_file.astype(np.float32)
    result = whisper.transcribe(_whisper_model, audio_file, temperature=0.0)
    return result["text"] or "NA"


# For testing purposes
if __name__ == "__main__":
    audio_file = sys.argv[1]
    result = whisper_ASR(audio_file)
    print(f'Whisper ASR Result: {result}')
