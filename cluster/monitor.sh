#!/usr/bin/env bash
# monitor.sh — live status for the dataset array job
# Usage:
#   bash monitor.sh              # one-shot snapshot
#   watch -n 60 bash monitor.sh  # refresh every 60 s

set -uo pipefail

SESSIONS_FILE="$(dirname "$0")/sessions.txt"
RAW_DIR="$HOME/analyzeMiceTrajectory/dataset/predictions_raw"
TRIALS_DIR="$HOME/analyzeMiceTrajectory/dataset"
GPU_COST_PER_HR=0.10   # L4 rate $/GPU-hr
JOB_NAME="dataset"     # matches -J "dataset[1-26]"

# ── helpers ────────────────────────────────────────────────────────────────
hr() { printf '%0.s─' $(seq 1 70); echo; }
bold() { printf '\033[1m%s\033[0m' "$1"; }

# ── 1. Job table ────────────────────────────────────────────────────────────
hr
bold "  JOB STATUS  (bjobs -J \"$JOB_NAME\")"
echo

# Collect running/pending array elements
bjobs -noheader -o 'jobid stat queue run_time' -J "$JOB_NAME" 2>/dev/null \
    | sort -k2,2 \
    | awk '{
        stat=$2; queue=$3; rt=$4;
        counts[stat]++;
        if (stat == "RUN") total_run_sec += rt+0;
        if (stat == "RUN" || stat == "PEND") active++;
    }
    END {
        printf "  %-10s %-10s %-10s %-10s\n", "RUN", "PEND", "DONE/EXIT", "Active GPUs"
        printf "  %-10s %-10s %-10s %-10s\n",
            counts["RUN"]+0, counts["PEND"]+0,
            counts["DONE"]+counts["EXIT"]+0,
            active+0
    }'

# Per-job detail (index, status, queue, elapsed)
echo
printf "  %-6s %-8s %-14s %-8s  %s\n" "IDX" "STAT" "QUEUE" "ELAPSED" "SESSION"
printf "  %-6s %-8s %-14s %-8s  %s\n" "---" "----" "-----" "-------" "-------"

mapfile -t SESSIONS < "$SESSIONS_FILE"

bjobs -noheader -o 'jobid stat queue run_time' -J "$JOB_NAME" 2>/dev/null \
    | sort -t'[' -k2 -V \
    | while IFS= read -r line; do
        jobid=$(echo "$line" | awk '{print $1}')
        stat=$(echo  "$line" | awk '{print $2}')
        queue=$(echo "$line" | awk '{print $3}')
        rt=$(echo   "$line"  | awk '{print $4}')

        # Extract array index from jobid like 12345[10]
        idx=$(echo "$jobid" | grep -oP '\[\K[0-9]+(?=\])')
        [ -z "$idx" ] && idx=$(echo "$jobid" | grep -oP '[0-9]+$')

        sess="${SESSIONS[$((idx-1))]:-?}"

        if [ "$rt" -gt 0 ] 2>/dev/null; then
            h=$((rt/3600)); m=$(((rt%3600)/60)); s=$((rt%60))
            elapsed=$(printf "%dh%02dm" $h $m)
        else
            elapsed="-"
        fi

        printf "  %-6s %-8s %-14s %-8s  %s\n" "$idx" "$stat" "$queue" "$elapsed" "$sess"
    done

# ── 2. Prediction progress ──────────────────────────────────────────────────
hr
bold "  PREDICTION PROGRESS"
echo

# Count expected trials per session from trials CSVs
declare -A expected
while IFS=, read -r animal vf _rest; do
    [[ "$animal" == "animal" ]] && continue
    key="$animal/$vf"
    expected["$key"]=$(( ${expected["$key"]:-0} + 1 ))
done < <(cat "$TRIALS_DIR"/rory_trials.csv "$TRIALS_DIR"/wilfred_trials.csv 2>/dev/null)

total_expected=0
total_done=0

printf "  %-40s  %6s / %-6s  %s\n" "SESSION" "DONE" "TOTAL" "BAR"
printf "  %-40s  %6s   %-6s\n" "-------" "----" "-----"

for sess in "${SESSIONS[@]}"; do
    [ -z "$sess" ] && continue
    animal=$(echo "$sess" | cut -d/ -f1)
    vf=$(echo "$sess"     | cut -d/ -f2)
    sess_dir="$RAW_DIR/${animal}_${vf}"

    exp=${expected["$sess"]:-0}
    done_n=0
    if [ -d "$sess_dir" ]; then
        done_n=$(find "$sess_dir" -name 'data3D.csv' 2>/dev/null | wc -l)
    fi

    total_expected=$((total_expected + exp))
    total_done=$((total_done + done_n))

    # Simple bar (20 chars)
    pct=0
    [ "$exp" -gt 0 ] && pct=$((done_n * 20 / exp))
    bar=$(printf '%0.s█' $(seq 1 $pct))
    bar=$(printf "%-20s" "$bar")

    printf "  %-40s  %6d / %-6d  [%s]\n" "$sess" "$done_n" "$exp" "$bar"
done

echo
pct_total=0
[ "$total_expected" -gt 0 ] && pct_total=$((total_done * 100 / total_expected))
printf "  TOTAL: %d / %d trials complete (%d%%)\n" \
    "$total_done" "$total_expected" "$pct_total"

# ── 3. Cost estimate ────────────────────────────────────────────────────────
hr
bold "  COST ESTIMATE  (L4 @ \$$GPU_COST_PER_HR / GPU-hr)"
echo

# Sum GPU-hours from all currently running jobs
total_gpu_hr=$(
    bjobs -noheader -o 'stat run_time' -J "$JOB_NAME" 2>/dev/null \
    | awk -v rate="$GPU_COST_PER_HR" '
        $1 == "RUN" { total_sec += $2+0 }
        END { printf "%.2f", total_sec / 3600 }
    '
)
cost_so_far=$(echo "$total_gpu_hr $GPU_COST_PER_HR" | awk '{printf "%.2f", $1 * $2}')

# Estimate remaining: 4.5M frames total, ~14 fps on L4
TOTAL_FRAMES=4516734
DONE_FRAMES=$(echo "$total_done $TOTAL_FRAMES $total_expected" \
    | awk '{if($3>0) printf "%d", $1/$3*$2; else print 0}')
REMAINING_FRAMES=$((TOTAL_FRAMES - DONE_FRAMES))
# Remaining wall time dominated by longest unfinished session
# Rough estimate: remaining_frames / 14fps / parallel_factor
n_active=$(bjobs -noheader -J "$JOB_NAME" 2>/dev/null | grep -c RUN || true)
[ "$n_active" -eq 0 ] && n_active=1
remaining_hr=$(echo "$REMAINING_FRAMES $n_active" \
    | awk '{fps=14; par=$2; printf "%.1f", $1/(fps*par*3600)}')
cost_remaining=$(echo "$remaining_hr $n_active $GPU_COST_PER_HR" \
    | awk '{printf "%.2f", $1 * $2 * $3}')

echo "  GPU-hours consumed so far : ${total_gpu_hr} hr  → \$${cost_so_far}"
echo "  Est. remaining wall time  : ~${remaining_hr} hr"
echo "  Est. remaining cost       : ~\$${cost_remaining}"

hr
echo "  Tip: run  tail -f ~/analyzeMiceTrajectory/cluster/logs/job_*_<IDX>.out"
echo "       to watch live output for a specific array element."
hr
