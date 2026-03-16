#!/bin/bash
# Submit the dataset generation array job to the cluster.
# Run this from the cluster login node:
#   bash ~/analyzeMiceTrajectory/cluster/submit.sh
#
# To submit only a subset of sessions (e.g. sessions 1-5):
#   bash ~/analyzeMiceTrajectory/cluster/submit.sh 1-5
#
# Session index → animal/video_folder mapping is in sessions.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LSF_SCRIPT="$SCRIPT_DIR/run_session.lsf"
LOG_DIR="$SCRIPT_DIR/logs"
SESSIONS_FILE="$SCRIPT_DIR/sessions.txt"

RANGE="${1:-1-26}"
START="${RANGE%%-*}"
END="${RANGE##*-}"

mkdir -p "$LOG_DIR"

echo "Sessions to submit (index $RANGE):"
awk -v s="$START" -v e="$END" 'NR>=s && NR<=e {printf "  [%d] %s\n", NR, $0}' "$SESSIONS_FILE"
echo ""
echo "Queue: gpu_a100  |  12 cores + 1 GPU per job  |  \$0.10/hr"
echo "Est. cost: ~\$$(echo "scale=0; ($END - $START + 1) * 8 / 10" | bc) (at 8 hrs/session)"
echo ""
read -p "Submit? [y/N] " confirm
[[ "$confirm" == "y" || "$confirm" == "Y" ]] || { echo "Aborted."; exit 0; }

JOB_ID=$(bsub -J "dataset[$RANGE]" < "$LSF_SCRIPT" | grep -oP '(?<=Job <)\d+')
echo ""
echo "Submitted job array $JOB_ID[$RANGE]"
echo ""
echo "Monitor:"
echo "  bjobs -u doq"
echo "  tail -f $LOG_DIR/job_${JOB_ID}_*.out"
echo "  grep 'Done\|FAILED\|fps' $LOG_DIR/job_${JOB_ID}_*.out | sort"
