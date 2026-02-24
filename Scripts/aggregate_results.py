"""
aggregate_results.py - Aggregate Harvard sentences experiment results.

Walks outputs/results/sentence_*/run_*/run_summary.json,
flattens each entry, and writes:
  - outputs/all_results.json  (list of full run_summary dicts)
  - outputs/all_results.csv   (one row per run, flattened)

Usage:
    python Scripts/aggregate_results.py
    python Scripts/aggregate_results.py --results_dir outputs/results --output_dir outputs
"""

import os
import json
import csv
import glob
import argparse


def aggregate_results(results_dir: str = "outputs/results", output_dir: str = "outputs"):
    pattern = os.path.join(results_dir, "*", "sentence_*", "run_*", "run_summary.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"No run_summary.json files found under {results_dir}")
        return []

    print(f"Found {len(json_files)} run summaries")

    all_summaries = []
    all_rows = []

    for json_path in json_files:
        with open(json_path, "r") as f:
            summary = json.load(f)
        all_summaries.append(summary)

        # Flatten to a single dict for CSV
        row = {}

        # metadata
        meta = summary.get("metadata", {})
        row["run_timestamp"] = meta.get("run_timestamp")
        row["sentence_id"] = meta.get("sentence_id")
        row["run_id"] = meta.get("run_id")
        row["timestamp"] = meta.get("timestamp")
        row["hardware"] = meta.get("hardware")

        # text
        text = summary.get("text_data", {})
        row["ground_truth_text"] = text.get("ground_truth_text")
        row["target_text"] = text.get("target_text")
        row["asr_transcription"] = text.get("asr_transcription")

        # success
        success = summary.get("success_metrics", {})
        row["success"] = success.get("success")
        for obj, score in success.get("fitness_scores", {}).items():
            row[f"score_{obj}"] = score
        for obj, threshold in success.get("thresholds", {}).items():
            row[f"threshold_{obj}"] = threshold

        # efficiency
        eff = summary.get("efficiency_metrics", {})
        row["generation_count"] = eff.get("generation_count")
        row["elapsed_time_seconds"] = eff.get("elapsed_time_seconds")
        row["avg_time_per_generation"] = eff.get("avg_time_per_generation")

        # algorithm parameters
        algo = summary.get("algorithm_parameters", {})
        row["attack_mode"] = algo.get("attack_mode")
        row["objectives"] = ",".join(algo.get("objectives", []))
        row["pop_size"] = algo.get("pop_size")
        row["num_generations"] = algo.get("num_generations")
        row["size_per_phoneme"] = algo.get("size_per_phoneme")
        row["iv_scalar"] = algo.get("iv_scalar")
        row["subspace_optimization"] = algo.get("subspace_optimization")

        # final solution
        sol = summary.get("final_solution", {})
        row["generation_found"] = sol.get("generation_found")

        row["pareto_front_size"] = len(summary.get("pareto_front", []))
        row["json_path"] = json_path

        all_rows.append(row)

    os.makedirs(output_dir, exist_ok=True)

    # Save all_results.json
    json_out = os.path.join(output_dir, "all_results.json")
    with open(json_out, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved {json_out}")

    # Save all_results.csv
    if all_rows:
        # Collect all fieldnames across all rows (union, preserving insertion order)
        fieldnames = list(dict.fromkeys(k for row in all_rows for k in row))
        csv_out = os.path.join(output_dir, "all_results.csv")
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {csv_out}")

    # Print summary statistics
    total = len(all_rows)
    successes = sum(1 for r in all_rows if r.get("success"))
    print(f"\n=== Summary ===")
    print(f"Total runs  : {total}")
    print(f"Successful  : {successes} ({100 * successes / total:.1f}%)")

    sentences = sorted(set(r["sentence_id"] for r in all_rows if r["sentence_id"] is not None))
    print(f"Sentences   : {len(sentences)} ({min(sentences)} – {max(sentences)})")

    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Harvard sentences experiment results.")
    parser.add_argument("--results_dir", default="outputs/results",
                        help="Root directory containing sentence_*/run_*/ subfolders")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory to write all_results.json and all_results.csv")
    args = parser.parse_args()
    aggregate_results(args.results_dir, args.output_dir)
