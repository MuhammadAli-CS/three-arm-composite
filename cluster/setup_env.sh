#!/usr/bin/env bash
# One-time environment setup for the three-arm experiment on Unicorn
# (it.coecis.cornell.edu).
#
# If you already have ~/morbo-env from the MOBO-HighDim repo, you do NOT need
# this -- that env already has torch/botorch/gpytorch and these jobs need
# nothing else.  Just activate it:
#
#   source /share/apps/software/anaconda3/etc/profile.d/conda.sh
#   conda activate ~/morbo-env
#
# Run this from an interactive allocation, NOT the login node -- the login
# node kills resource-intensive processes, and a pip install of torch counts.
#
#   salloc --mem=8g --cpus-per-task=4 --time=01:00:00 \
#          --partition=default_partition-interactive --account=kilian
#   bash cluster/setup_env.sh
set -euo pipefail

ENV_PATH="${ENV_PATH:-$HOME/three-arm-env}"

if [ ! -d "/share/apps/software/anaconda3" ]; then
  echo "WARNING: expected conda install at /share/apps/software/anaconda3 not found." >&2
  echo "Check 'module avail' or the Unicorn docs -- the path may have changed." >&2
fi

# conda init only takes effect in a NEW interactive shell (it edits ~/.bashrc,
# which is guarded behind an "if not interactive, return" near the top), and
# this script is a non-interactive subshell -- so source conda's own profile
# script directly instead, which works either way.
source /share/apps/software/anaconda3/etc/profile.d/conda.sh

if conda env list | grep -q "$ENV_PATH"; then
  echo "Env already exists at $ENV_PATH -- skipping creation."
else
  conda create -y -p "$ENV_PATH" python=3.11
fi
conda activate "$ENV_PATH"

pip install --upgrade pip

# CPU-only torch, deliberately.  three_arm.sub requests --cpus-per-task=1 and
# no GPU: these runs are Cholesky factorisations of matrices up to 45x45, where
# a GPU buys nothing and only lengthens the queue wait.  The CPU wheel is a few
# hundred MB against ~2.5 GB for cu128, which is most of the difference in how
# long this script takes.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# botorch/gpytorch are pinned to what the stored results were produced with.
# The code calls APIs that moved after 0.9.5, so "just use latest" breaks it.
pip install "gpytorch==1.11" "botorch==0.9.5" numpy matplotlib

# No `pip install -e .`: this repo is a plain directory, not a package, and
# three_arm/run.py puts the repo root on sys.path itself.
python - <<'PYEOF'
import botorch, gpytorch, torch
assert botorch.__version__.startswith("0.9.5"), botorch.__version__
assert gpytorch.__version__.startswith("1.11"), gpytorch.__version__
print(f"OK: torch={torch.__version__} botorch={botorch.__version__} "
      f"gpytorch={gpytorch.__version__}")
PYEOF

echo
echo "Done.  Submit with:"
echo "  ENV_PATH=$ENV_PATH bash cluster/submit_three_arm.sh validate sweep pairs bijectivity"
