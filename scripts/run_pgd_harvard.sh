#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# Split attack ranges
START=1
END=100

# ---------- Generate all audios first ----------
python scripts/generate_harvard_audios.py --start $START --end $END

for i in $(seq $START $END); do
    python scripts/generate_harvard_audios.py --start $i --end $i

    conda run --no-capture-output -n pgd python scripts/adversarial_pgd_harvard.py \
        --start $i --end $i \
        --gpu 0
done