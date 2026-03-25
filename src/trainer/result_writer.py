import os
import json

import torch
import soundfile as sf


def save_attack_result(
    sentence_id: int,
    method: str,
    audio,
    transcription: str,
    gt_text: str,
    elapsed: float,
    params: dict,
) -> None:
    """
    Save adversarial audio and a standardised JSON for one attack result.

    Args:
        sentence_id:   1-based Harvard sentence index.
        method:        One of 'tts', 'waveform', 'smack', 'pgd'.
        audio:         Audio as a numpy array or torch.Tensor.
        transcription: Whisper transcription from optimization time.
        gt_text:       Original Harvard sentence text.
        elapsed:       Wall-clock seconds the attack took.
        params:        Method-specific hyperparameters dict.
    """
    sentence_dir = os.path.join('outputs', f'harvard_sentence_{sentence_id:03d}')
    os.makedirs(sentence_dir, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy().squeeze()

    sf.write(os.path.join(sentence_dir, f'{method}.wav'), audio, samplerate=16000)

    with open(os.path.join(sentence_dir, f'{method}.json'), 'w') as f:
        json.dump({
            'method': method,
            'sentence_id': sentence_id,
            'gt_text': gt_text,
            'transcription': transcription,
            'elapsed_seconds': round(elapsed, 2),
            'params': params,
        }, f, indent=2)

    print(f"[{sentence_id:3d}] Saved {method}.wav | transcription: {transcription!r}")