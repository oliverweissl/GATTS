#!/bin/bash
set -e

conda env create -f "$(dirname "$0")/pgd.yml"
mkdir -p "$(dirname "$0")/../outputs/test-clean"
conda run -n pgd pip install setuptools==65.5.0
conda run -n pgd pip install --no-deps --no-build-isolation \
    git+https://github.com/RaphaelOlivier/pyaudlib.git \
    git+https://github.com/RaphaelOlivier/robust_speech.git \
    git+https://github.com/RaphaelOlivier/speechbrain.git \
