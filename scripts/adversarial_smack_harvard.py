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
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SMACK"))

import time
import argparse
import torch
import soundfile as sf
import librosa


from src.data.harvard_sentences import HARVARD_SENTENCES
from src.models._whisper import Whisper
from src.trainer.result_writer import save_attack_result
from scripts.SMACK.genetic import GeneticAlgorithm
from scripts.SMACK.gradient import GradientEstimation
from scripts.SMACK.synthesis import audio_synthesis


POPULATION_SIZE = 163
GENETIC_ITERATIONS = 82
GRADIENT_ITERATIONS = 41
TARGET_MODEL = 'whisperASR'

AUDIO_DIR = 'outputs'


def run_attack(reference_audio: str, reference_text: str):
    """Run genetic + gradient attack for one sentence."""
    start_time = time.time()

    ga = GeneticAlgorithm(reference_audio, reference_text, TARGET_MODEL, POPULATION_SIZE)
    fittest_individual = ga.run(GENETIC_ITERATIONS)

    print("Genetic algorithm finished. Launching gradient estimation.\n")

    gradient_estimator = GradientEstimation(
        reference_audio, reference_text, TARGET_MODEL,
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
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    print(f"Sentences: {args.start} → {args.end}")
    print(f"Population: {POPULATION_SIZE} | Genetic: {GENETIC_ITERATIONS} | Gradient: {GRADIENT_ITERATIONS}")
    print(f"Audio directory: {AUDIO_DIR}")
    print('=' * 60)

    whisper_model = Whisper()

    if not os.path.exists("scripts/SMACK/SampleDir"):
        os.mkdir("scripts/SMACK/SampleDir")

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = HARVARD_SENTENCES[sentence_id - 1]

        sentence_dir = os.path.join(AUDIO_DIR, f'harvard_sentence_{sentence_id:03d}')
        reference_audio = os.path.join(sentence_dir, 'harvard_audio.wav')

        if not os.path.exists(reference_audio):
            print(f"[{sentence_id:3d}] Reference audio not found, skipping: {reference_audio}")
            continue

        print(f"\n{'='*60}")
        print(f"[Sentence {sentence_id}] {sentence_text}")
        print('=' * 60)

        p_refined, elapsed = run_attack(reference_audio, sentence_text)

        # Synthesize adversarial audio from the refined prosody vector
        # ETTS (WaveGlow) outputs at 22050 Hz — resample to 16 kHz for consistency
        audio_numpy = audio_synthesis(p_refined.reshape(-1, 32), reference_audio, sentence_text)
        audio_16k = librosa.resample(audio_numpy.astype('float32'), orig_sr=22050, target_sr=16000)

        texts, _ = whisper_model.inference(torch.from_numpy(audio_16k).float())
        transcription = texts[0]

        save_attack_result(
            sentence_id=sentence_id,
            method='smack',
            audio=audio_16k,
            transcription=transcription,
            gt_text=sentence_text,
            elapsed=elapsed,
            params={
                'num_generations': GENETIC_ITERATIONS + GRADIENT_ITERATIONS,
                'pop_size': POPULATION_SIZE,
                'target_model': TARGET_MODEL,
                'genetic_iterations': GENETIC_ITERATIONS,
                'gradient_iterations': GRADIENT_ITERATIONS,
            },
        )

    print("\n[Done]")


if __name__ == '__main__':
    main()