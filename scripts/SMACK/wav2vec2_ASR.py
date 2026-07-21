import os
import sys
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models import Wav2Vec2ASR

_SYNTHESIS_SAMPLE_RATE = 22_050
_wav2vec2_model = None


def _load_model():
    global _wav2vec2_model
    if _wav2vec2_model is None:
        _wav2vec2_model = Wav2Vec2ASR()


def wav2vec2_ASR(audio_file: Any) -> str:
    _load_model()
    sample_rate = _SYNTHESIS_SAMPLE_RATE if isinstance(audio_file, np.ndarray) else 16_000
    texts, _ = _wav2vec2_model.inference(audio_file, sample_rate=sample_rate)
    return texts[0] if texts and texts[0] else "NA"


if __name__ == "__main__":
    audio_file = sys.argv[1]
    result = wav2vec2_ASR(audio_file)
    print(f"Wav2Vec2 ASR Result: {result}")
