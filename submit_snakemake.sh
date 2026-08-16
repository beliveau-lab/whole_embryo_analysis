#!/bin/bash
#
# Submit the pipeline for one or more runs.
#
#   ./submit_snakemake.sh 306
#   ./submit_snakemake.sh 306 307 308
#   ./submit_snakemake.sh 306 -- --rerun-incomplete --dry-run
#
# Each run gets its own working directory, so concurrent submissions do not
# fight over snakemake's directory lock.

set -euo pipefail

PROJ="$HOME/msf_project/whole_embryo_analysis2"
JOB="$PROJ/job.sh"

# Split args: run ids before --, snakemake passthrough after.
RUNS=()
EXTRA=()
seen_sep=0
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then seen_sep=1; continue; fi
    if [[ $seen_sep -eq 1 ]]; then EXTRA+=("$arg"); else RUNS+=("$arg"); fi
done

# done: (300 306 324 308 309_retry 321_retry 312 299

if [[ ${#RUNS[@]} -eq 0 ]]; then
    RUNS=(300 306 324 308 309_retry 321_retry 312)
fi

[[ -f "$JOB" ]] || { echo "ERROR: missing $JOB" >&2; exit 1; }

for RUN in "${RUNS[@]}"; do
    CONFIG="$PROJ/configs/${RUN}.yaml"
    LOGS="$PROJ/logs/${RUN}"

    if [[ ! -f "$CONFIG" ]]; then
        echo "SKIP $RUN: no $CONFIG" >&2
        continue
    fi

    # SGE will not create the -o directory for you; the job dies if it is absent.
    mkdir -p "$LOGS"

    # -o is set here rather than via #$ in job.sh because qsub does not expand
    # shell variables in directives. Command-line options override #$ lines.
    # \$JOB_ID is left unexpanded for SGE to fill in.
    qsub -N "run${RUN}" \
         -l m_mem_free=64G \
         -l h_rt=72:00:00 \
         -pe serial 4 \
         -v RUN="$RUN" \
         -o "$LOGS/snakemake.\$JOB_ID.log" \
         "$JOB" "${EXTRA[@]}"
done

echo
echo "Submitted. Check with: qstat -u $USER"
echo "Logs: $PROJ/logs/<run>/"


