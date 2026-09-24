#!/usr/bin/env bash
# One-time environment setup on the Unicorn cluster (it.coecis.cornell.edu).
# Run this from an interactive allocation, NOT the login node — the login
# node kills resource-intensive processes automatically, and building a
# conda env / installing torch counts as one.
#
# Usage (from unicorn-login-01):
#   salloc --mem=8g --cpus-per-task=4 --time=01:00:00 --partition=default_partition-interactive --account=kilian
#   bash cluster/setup_env.sh
set -euo pipefail

ENV_PATH="$HOME/morbo-env"

if [ ! -d "/share/apps/software/anaconda3" ]; then
  echo "WARNING: expected conda install at /share/apps/software/anaconda3 not found." >&2
  echo "Check 'module avail' or the Unicorn docs — path may have changed." >&2
fi

# Only needs to run once per account; harmless to re-run. `conda init bash`
# only takes effect in a NEW interactive shell (it edits ~/.bashrc, which
# most distros guard behind an "if not interactive, return" check near the
# top) -- this script runs as a non-interactive subshell, so `source
# ~/.bashrc` here doesn't actually pull conda's `conda`/`activate` shell
# functions into scope. Source conda's own profile script directly instead,
# which works regardless of interactive/non-interactive shell state.
/share/apps/software/anaconda3/bin/conda init bash || true
source /share/apps/software/anaconda3/etc/profile.d/conda.sh

if conda env list | grep -q "$ENV_PATH"; then
  echo "Env already exists at $ENV_PATH — skipping creation."
else
  conda create -y -p "$ENV_PATH" python=3.11
fi

conda activate "$ENV_PATH"

# torch: pinned to the cu128 index (CUDA 12.8), matching the aimi/gpu
# partition nodes' actual NVIDIA driver ("found version 12080" from
# torch.cuda.is_available()'s warning -- drivers are backward- but not
# forward-compatible, so a wheel built for a newer CUDA than the driver
# supports silently reports cuda unavailable and falls back to CPU rather
# than erroring loudly). Confirmed on the cluster: unpinned pip installed
# 2.13.0+cu130 (built for CUDA 13.0), which torch.cuda.is_available()
# silently rejected -- three jobs ran a while on CPU before this was caught.
# Plain PyPI's default wheel and the old cu121 channel (frozen at <=2.5.1,
# and CUDA 12.1 predates Blackwell/compute-capability-10.0 support anyway)
# are both wrong for this hardware; cu128 is the one that actually matches.
#
# botorch/gpytorch: DELIBERATELY pinned to the exact versions this codebase
# was ported against (see README.md "Fork notes" -- the port removed/renamed
# botorch APIs that changed between the archived upstream's target, ~0.6,
# and 0.9.5). "Just use latest" risks botorch having moved its API again
# since 0.9.5, silently breaking the same way the original archived-repo
# port had to fix -- pinning trades a possibly-newer botorch for guaranteed
# API compatibility with the code as actually written.
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "gpytorch==1.11" "botorch==0.9.5" scipy matplotlib jupyter anthropic

# Install this repo (morbo/, botier_llm/, run_comparison.py, etc.) in editable mode.
cd "$(dirname "$0")/.."
pip install -e .

# OPTIONAL: LassoBench, needed only for the evalfn="LassoBenchMO"
# experiments (lasso_*_mo). Installed from source (not on PyPI); its deps
# (celer, sparse-ho) compile C extensions, which is why it's opt-in rather
# than a hard requirement of this repo. submit_real_benchmarks.sh checks
# for it and refuses to submit LassoBench jobs if missing.
if python -c "import LassoBench" 2>/dev/null; then
  echo "LassoBench already installed."
else
  echo "Installing LassoBench (optional; needed for lasso_*_mo experiments)..."
  if [ ! -d "$HOME/LassoBench" ]; then
    git clone https://github.com/ksehic/LassoBench.git "$HOME/LassoBench" || true
  fi
  pip install -e "$HOME/LassoBench" || {
    echo "WARNING: LassoBench install failed -- lasso_*_mo experiments will" >&2
    echo "not run until it's installed manually. Everything else is unaffected." >&2
  }
  # CRITICAL: LassoBench's dependency resolution can silently upgrade
  # botorch/gpytorch past the pinned versions (confirmed on the cluster:
  # after installing LassoBench, `botorch.sampling.deterministic` no longer
  # existed and every job died on import). Force the pins back afterward,
  # unconditionally.
  pip install "gpytorch==1.11" "botorch==0.9.5"
  python - <<'PYEOF'
import botorch, gpytorch, torch
from botorch.sampling.deterministic import DeterministicSampler  # the import that broke
assert botorch.__version__.startswith("0.9.5"), botorch.__version__
assert gpytorch.__version__.startswith("1.11"), gpytorch.__version__
print(f"pins OK: botorch={botorch.__version__} gpytorch={gpytorch.__version__} "
      f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
PYEOF
fi

echo "Done. Activate with: conda activate $ENV_PATH"
