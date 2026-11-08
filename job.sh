# Runs the whole_embryo_analysis pipeline for one config.
# Do not submit this directly -- use submit_snakemake.sh, which creates the
# log directory and passes RUN. Direct qsub works but you get RUN=306 and
# the scheduler's default output path.
#
# Options after -- are forwarded to snakemake, e.g.
#   ./submit_snakemake.sh 306 -- --rerun-incomplete

set -euo pipefail

# ── environment ─────────────────────────────────────────────────────────────
conda activate /net/beliveau/vol1/home/msforman/.conda/envs/visualization
echo "conda activated"

# ── paths ───────────────────────────────────────────────────────────────────
PROJ="/net/beliveau/vol1/project/whole_embryo_analysis/whole_embryo_analysis"
cd "$PROJ"

RUN="${RUN:-306}"
CONFIG="$PROJ/configs/${RUN}.yaml"
LOGS="$PROJ/logs/${RUN}"
WORK="$PROJ/runs/${RUN}"          # isolated .snakemake/ so runs can overlap

mkdir -p "$LOGS" "$WORK"

# ── preflight ───────────────────────────────────────────────────────────────
# Fail here with a clear message 
command -v python >/dev/null || { echo "ERROR: python not on PATH after activate" >&2; exit 1; }
command -v snakemake >/dev/null || { echo "ERROR: snakemake not on PATH" >&2; exit 1; }
python -c "import polars" || { echo "ERROR: polars missing from $CONDA_ENV" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: missing $CONFIG" >&2; exit 1; }
[[ -f "$PROJ/Snakefile" ]] || { echo "ERROR: missing $PROJ/Snakefile" >&2; exit 1; }

echo "RUN=$RUN"
echo "CONFIG=$CONFIG"
echo "WORK=$WORK"
echo "LOGS=$LOGS"
echo "HOST=$(hostname)  JOB_ID=${JOB_ID:-none}"

# ── run ─────────────────────────────────────────────────────────────────────
# --snakefile and --configfile must be absolute: with --directory set,
# snakemake resolves relative paths against WORK, not PROJ.
# log_dir is passed so rule logs land in PROJ/logs/RUN, not WORK/logs/RUN.
snakemake -p \
    --snakefile "$PROJ/Snakefile" \
    --directory "$WORK" \
    --configfile "$CONFIG" \
    --config run_id="$RUN" log_dir="$LOGS" \
    --cores 1 \
    --until somite_clustering \
    "$@"

# dry run
# snakemake -n -p \
#     --snakefile "$PROJ/Snakefile" \
#     --directory "$WORK" \
#     --configfile "$CONFIG" \
#     --config run_id="$RUN" log_dir="$LOGS" \
#     --cores 1 \
#     --until somite_clustering \
#     "$@"
