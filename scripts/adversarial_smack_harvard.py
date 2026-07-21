"""
Adversarial SMACK — Harvard Sentences Experiment

Loads pre-generated reference audio from HarvardAudios/ and runs the SMACK
untargeted attack (genetic + gradient) on each Harvard sentence.

Generate reference audios first (styletts2 env):
    python scripts/generate_harvard_audios.py --start 1 --end 100

Then run this script (smack env) from the SMACK directory:
    python adversarial_smack_harvard.py --start 1 --end 10
"""

import os
import random

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SMACK"))

import time
import argparse
import datetime
import soundfile as sf
from typing import Optional


from src.data.harvard_sentences import HARVARD_SENTENCES
from scripts.SMACK.genetic import ASR_DISPATCH, GeneticAlgorithm
from scripts.SMACK.gradient import GradientEstimation
from scripts.SMACK.synthesis import audio_synthesis
from src.trainer.attack_summary import compute_attack_summary


POPULATION_SIZE = 20
GENETIC_ITERATIONS = 10
GRADIENT_ITERATIONS = 5
ASR_MODEL_CHOICES = ("whisper", "wav2vec2", "speechbrain")
_ASR_MODEL_MAP = {
    "whisper": "whisperASR",
    "whisperasr": "whisperASR",
    "wav2vec2": "wav2vec2ASR",
    "wav2vec2asr": "wav2vec2ASR",
    "speechbrain": "speechbrainASR",
    "speechbrainasr": "speechbrainASR",
}

AUDIO_DIR = 'outputs/'


def resolve_smack_asr_model(name: str) -> str:
    key = name.replace("_", "").replace("-", "").lower()
    try:
        return _ASR_MODEL_MAP[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported ASR model '{name}'. Choose one of: {', '.join(ASR_MODEL_CHOICES)}") from exc


def run_attack(reference_audio: str, reference_text: str, output_dir: str, target_text: Optional[str], target_model: str):
    """Run genetic + gradient attack for one sentence."""
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    ga = GeneticAlgorithm(reference_audio, reference_text, target_model, target_text, POPULATION_SIZE)
    fittest_individual = ga.run(GENETIC_ITERATIONS)

    print("Genetic algorithm finished. Launching gradient estimation.\n")

    gradient_estimator = GradientEstimation(
        reference_audio, reference_text, target_model, target_text,
        sigma=0.1, learning_rate=0.01, K=20
    )
    p_refined = gradient_estimator.refine_prosody_vector(fittest_individual, GRADIENT_ITERATIONS)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Attack finished. Time: {hours}h {minutes}m {seconds:.2f}s\n")

    return p_refined, elapsed


def main():
    parser = argparse.ArgumentParser(description='SMACK Untargeted Attack — Harvard Sentences')
    parser.add_argument('--start', type=int, default=1,
                        help='First Harvard sentence index (1-based)')
    parser.add_argument('--end', type=int, default=10,
                        help='Last Harvard sentence index (1-based, inclusive)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    parser.add_argument("--untargeted", action="store_true")
    parser.add_argument("--asr_model", type=str, default="whisper", choices=ASR_MODEL_CHOICES,
                        help="ASR backend for SMACK fitness evaluation")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    target_model = resolve_smack_asr_model(args.asr_model)

    print(f"Sentences: {args.start} → {args.end}")
    print(f"Population: {POPULATION_SIZE} | Genetic: {GENETIC_ITERATIONS} | Gradient: {GRADIENT_ITERATIONS}")
    print(f"ASR model: {args.asr_model} ({target_model})")
    print(f"Audio directory: {AUDIO_DIR}")
    print('=' * 60)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_base = os.path.join('outputs', 'results', 'SMACK_' + args.asr_model + ("" if args.untargeted else "_targeted"), run_timestamp)

    if not os.path.exists("scripts/SMACK/SampleDir"):
        os.mkdir("scripts/SMACK/SampleDir")

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = HARVARD_SENTENCES[sentence_id - 1]
        reference_audio = os.path.join(AUDIO_DIR,f'harvard_sentence_{sentence_id:03d}', 'harvard_audio.wav')

        if not os.path.exists(reference_audio):
            print(f"[{sentence_id:3d}] Reference audio not found, skipping: {reference_audio}")
            continue

        print(f"\n{'='*60}")
        print(f"[Sentence {sentence_id}] {sentence_text}")
        print('=' * 60)

        sentence_dir = os.path.join(output_base, f'sentence_{sentence_id:03d}')
        os.makedirs(sentence_dir, exist_ok=True)

        target_sentence = None if args.untargeted else random.choice([HARVARD_SENTENCES[i] for i in range(len(HARVARD_SENTENCES)) if i-1 != sentence_id])

        p_refined, elapsed = run_attack(reference_audio, sentence_text, sentence_dir, target_sentence, target_model)

        # Synthesize adversarial audio from the refined prosody vector
        # ETTS (WaveGlow) outputs at 22050 Hz — resample to 16 kHz for consistency
        import librosa
        audio_numpy = audio_synthesis(p_refined.reshape(-1, 32), reference_audio, sentence_text)
        audio_16k = librosa.resample(audio_numpy.astype('float32'), orig_sr=22050, target_sr=16000)
        adv_path = os.path.join(sentence_dir, 'best_smack.wav')
        gt_dst    = os.path.join(sentence_dir, 'ground_truth.wav')
        sf.write(adv_path, audio_16k, 16000)

        import shutil
        shutil.copy(reference_audio, gt_dst)

        summary_asr = ASR_DISPATCH[target_model]
        adv_transcription = summary_asr(adv_path)
        gt_transcription = summary_asr(gt_dst)

        compute_attack_summary(
            adversarial_audio_path=adv_path,
            gt_audio_path=gt_dst,
            gt_text=sentence_text,
            target_text=target_sentence,
            attack_method='SMACK',
            num_generations=GENETIC_ITERATIONS + GRADIENT_ITERATIONS,
            pop_size=POPULATION_SIZE,
            elapsed_time_seconds=elapsed,
            output_path=os.path.join(sentence_dir, 'smack_summary.json'),
            sentence_id=sentence_id,
            whisper_transcription=adv_transcription,
            gt_transcription=gt_transcription,
            extra={
                'asr_model': args.asr_model,
                'smack_target_model': target_model,
            },
        )

    print("\n[Done]")


if __name__ == '__main__':
    main()