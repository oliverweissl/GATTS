import os
import sys
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models import SpeechBrainASR

_SYNTHESIS_SAMPLE_RATE = 22_050
_speechbrain_model = None


def _load_model():
    global _speechbrain_model
    if _speechbrain_model is None:
        _speechbrain_model = SpeechBrainASR()


def speechbrain_ASR(audio_file: Any) -> str:
    _load_model()
    sample_rate = _SYNTHESIS_SAMPLE_RATE if isinstance(audio_file, np.ndarray) else 16_000
    texts, _ = _speechbrain_model.inference(audio_file, sample_rate=sample_rate)
    return texts[0] if texts and texts[0] else "NA"


if __name__ == "__main__":
    audio_file = sys.argv[1]
    result = speechbrain_ASR(audio_file)
    print(f"SpeechBrain ASR Result: {result}")
