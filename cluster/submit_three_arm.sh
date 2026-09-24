#!/usr/bin/env bash
# Three-arm composite experiment (knowing g vs observing h): one job per outer
# map, each running all seeds and all three arms.
#
# The map list is read from three_arm.test_functions.GROUPS so it cannot drift
# out of sync with the code.
#
# Usage:
#   bash cluster/submit_three_arm.sh              # every group
#   bash cluster/submit_three_arm.sh sweep        # one group
#   SEEDS=20 BUDGET=40 bash cluster/submit_three_arm.sh pairs
#   DRAWS="0 1 2 3" SEEDS=5 bash cluster/submit_three_arm.sh   # 4 functions x 5 designs
#
# DRAWS varies the random function h itself, not just the initial design, so a
# conclusion cannot be a property of one sampled function.  Total runs per map
# is (number of draws) x SEEDS; --collect pools them.
#
# Validate the arm-3 sampler before spending a queue on it:
#   python three_arm/run.py --check
#
# Collect the results once the jobs land:
#   python three_arm/run.py --collect
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs three_arm/results

SEEDS="${SEEDS:-10}"
BUDGET="${BUDGET:-40}"
ENV_PATH="${ENV_PATH:-$HOME/morbo-env}"
DRAWS="${DRAWS:-0}"
GROUPS_TO_RUN=("$@")
if [ ${#GROUPS_TO_RUN[@]} -eq 0 ]; then
  GROUPS_TO_RUN=(validate sweep pairs analogs)
fi

if [ ! -d "$ENV_PATH" ]; then
  echo "conda env not found at $ENV_PATH." >&2
  echo "Create it once with:      bash cluster/setup_env.sh" >&2
  echo "Or point at another one:  ENV_PATH=/path/to/env bash $0 $*" >&2
  exit 1
fi

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

# Preflight and map expansion in one call: fail here on the login node rather
# than discovering a missing package in all 19 queued jobs.  The map list is
# read from GROUPS so it cannot drift out of sync with the code, and is
# de-duplicated because maps appear in more than one group (shiftsq_c0 is in
# both the sweep and the matched pairs).
MAPS=$(python - "${GROUPS_TO_RUN[@]}" <<'PY'
import importlib.util, sys
missing = [m for m in ("torch", "botorch", "gpytorch")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing packages in the conda env: " + ", ".join(missing)
             + "\ninstall with:  pip install " + " ".join(missing))
try:
    from three_arm.test_functions import GROUPS
except ImportError as exc:
    sys.exit(f"cannot import three_arm ({exc}) -- is the repo checked out here?")
bad = [g for g in sys.argv[1:] if g not in GROUPS]
if bad:
    sys.exit(f"unknown group(s) {bad}; choose from {sorted(GROUPS)}")
print(" ".join(dict.fromkeys(m for g in sys.argv[1:] for m in GROUPS[g])))
PY
) || exit 1

echo "submitting $(echo "$MAPS" | wc -w) jobs (seeds=$SEEDS budget=$BUDGET): $MAPS"
for map in $MAPS; do
  sbatch --requeue \
    --job-name="three-arm-${map}" \
    --export=MAP="$map",SEEDS="$SEEDS",BUDGET="$BUDGET",ENV_PATH="$ENV_PATH" \
    --parsable \
    cluster/three_arm.sub
done
