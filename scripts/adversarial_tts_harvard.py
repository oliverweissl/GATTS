"""
Adversarial TTS — Harvard Sentences Experiment

Usage:
    python scripts/adversarial_tts_harvard.py [args]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from src.data.dataclass import ModelData
from src.trainer.environment_loader import EnvironmentLoader
from src.trainer.adversarial_trainer import AdversarialTrainer
from src.trainer.run_logger import RunLogger
from src.trainer.result_writer import save_attack_result
from src.trainer.vector_manipulator import VectorManipulator
from src.optimizer.pymoo_optimizer import PymooOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2
from src.data.enum import AttackMode

from src.data.harvard_sentences import HARVARD_SENTENCES


def initialize_parser():
    parser = argparse.ArgumentParser(description="Adversarial TTS — Harvard Sentences")
    parser.add_argument("--harvard_sentences_start", type=int, default=1)
    parser.add_argument("--harvard_sentences_end", type=int, default=10)
    parser.add_argument("--num_generations", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--size_per_phoneme", type=int, default=1)
    parser.add_argument("--seed_target", action="store_true", default=False)
    parser.add_argument("--seed_gt", action="store_true", default=False)
    parser.add_argument("--min_generations", type=int, default=0)
    parser.add_argument("--mode", type=str, default="NOISE_UNTARGETED")
    parser.add_argument("--target_text", type=str, default="")
    parser.add_argument("--objectives", type=str, default="PESQ=0.2, SET_OVERLAP=0.5")
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index to use (default: 0)')
    return parser


def main():
    parser = initialize_parser()
    args = parser.parse_args()

    try:
        mode = AttackMode[args.mode]
    except KeyError:
        raise ValueError(f"Invalid mode '{args.mode}'. Available modes: {[m.name for m in AttackMode]}")

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    loader = EnvironmentLoader(device)
    tts_model, asr_model = loader.load_required_models()
    print("Models loaded.")

    objectives, thresholds = loader.parse_objectives(args.objectives)

    print(f"{'='*60}")
    print(f"  mode:               {mode}")
    print(f"  sentences:          {args.harvard_sentences_start} → {args.harvard_sentences_end}")
    print(f"  generations:        {args.num_generations}")
    print(f"  min_generations:    {args.min_generations}")
    print(f"  pop_size:           {args.pop_size}")
    print(f"  batch_size:         {args.batch_size}")
    print(f"  objectives:         {args.objectives}")
    print(f"  size_per_phoneme:   {args.size_per_phoneme}")
    print(f"{'='*60}")

    try:
        for sentence_id in range(args.harvard_sentences_start, args.harvard_sentences_end + 1):
            print(f"\n{'='*60}")
            ground_truth_text = HARVARD_SENTENCES[sentence_id - 1]
            print(f"[Sentence {sentence_id}] {ground_truth_text}")
            target_text = args.target_text if args.target_text else HARVARD_SENTENCES[sentence_id % len(HARVARD_SENTENCES)]
            if mode is AttackMode.TARGETED:
                print(f"[Target]     {target_text}")
            print(f"{'='*60}")

            # Load pre-generated GT audio and embeddings
            sentence_dir = os.path.join('outputs', f'harvard_sentence_{sentence_id:03d}')
            gt_audio_path = os.path.join(sentence_dir, 'harvard_audio.wav')
            embeddings_path = os.path.join(sentence_dir, 'harvard_audio.pt')
            if not os.path.exists(gt_audio_path) or not os.path.exists(embeddings_path):
                print(f"[{sentence_id:3d}] GT audio or embeddings not found, skipping")
                continue
            audio_gt = torch.from_numpy(sf.read(gt_audio_path)[0]).float().to(device)
            audio_embedding_gt = torch.load(embeddings_path, map_location=device)

            # Generate target embedding
            audio_embedding_target = loader.generate_target_embedding(
                mode, target_text, audio_embedding_gt, tts_model
            )

            objectives_dict = loader.initialize_objectives(
                active_objectives=objectives,
                model_data=ModelData(tts_model=tts_model, asr_model=asr_model),
                text_gt=ground_truth_text,
                text_target=target_text,
                mode=mode,
                audio_gt=audio_gt,
            )

            vector_manipulator = VectorManipulator(audio_embedding_gt, audio_embedding_target.h_text, mode)
            trainer = AdversarialTrainer(tts_model, asr_model, thresholds, objectives_dict, vector_manipulator, device)
            logger = RunLogger(objectives, tts_model, asr_model, vector_manipulator, device)

            solution_shape = (int(audio_embedding_gt.input_length.detach().cpu().item()), args.size_per_phoneme)
            optimizer = PymooOptimizer(
                bounds=(0, 1),
                algorithm=NSGA2,
                algo_params={"pop_size": args.pop_size},
                num_objectives=len(objectives),
                solution_shape=solution_shape,
            )

            if args.seed_target or args.seed_gt:
                n_var = int(np.prod(solution_shape))
                initial_pop = np.random.uniform(0, 1, (args.pop_size, n_var))
                if args.seed_target:
                    initial_pop[0] = 1.0
                if args.seed_gt:
                    initial_pop[1] = 0.0
                optimizer.update_problem(solution_shape, sampling=initial_pop)

            fitness_data, archive_data, generation_count, elapsed_time_total, interrupted, generation_found = trainer.run_full_iteration(
                optimizer, args.num_generations, args.pop_size, args.batch_size, args.min_generations
            )

            if fitness_data:
                # Store Fitness Data
                obj_names = [obj.name for obj in objectives]
                rows = []
                for gen_idx, gen_matrix in enumerate(fitness_data):
                    df_gen = pd.DataFrame(gen_matrix, columns=obj_names)
                    df_gen['generation'] = gen_idx + 1
                    df_gen['individual_id'] = df_gen.index
                    rows.append(df_gen)
                df = pd.concat(rows, ignore_index=True)[['generation', 'individual_id'] + obj_names]
                df.to_csv(os.path.join(sentence_dir, 'tts_fitness_history.csv'), index=False)

                # Select Best Candidate
                best_candidate = logger.select_best_candidate(optimizer.best_candidates, thresholds)
                audio_best, text_best = logger.run_final_inference(best_candidate)

                # Store tts.wav and tts.json
                save_attack_result(
                    sentence_id=sentence_id,
                    method='tts',
                    audio=audio_best,
                    transcription=text_best,
                    gt_text=ground_truth_text,
                    elapsed=elapsed_time_total,
                    params={
                        'num_generations': args.num_generations,
                        'generations_run': generation_count,
                        'generation_found': generation_found,
                        'pop_size': args.pop_size,
                        'batch_size': args.batch_size,
                        'mode': args.mode,
                        'objectives': args.objectives,
                        'size_per_phoneme': args.size_per_phoneme,
                    },
                )

            torch.cuda.empty_cache()
            if interrupted:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[!] Experiment stopped early. All completed runs have been saved.")

    print("\n[Done]")


if __name__ == "__main__":
    main()
