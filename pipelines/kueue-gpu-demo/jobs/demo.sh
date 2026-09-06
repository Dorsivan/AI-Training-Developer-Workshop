#!/usr/bin/env bash
#
# Kueue GPU Preemption & Re-admission Demo
#
# Demonstrates with batch/v1 Jobs (not bare pods) that Kueue can:
#   1. Admit jobs using own quota
#   2. Admit jobs using borrowed quota (cohort)
#   3. Preempt a borrowed job when the lending queue needs its GPU back
#   4. Automatically re-admit the preempted job when a GPU frees up
#
# Cluster setup (already in place):
#   - 2 physical GPUs
#   - team-a-cluster-queue: nominalQuota=1, borrowingLimit=1, lendingLimit=1
#   - team-b-cluster-queue: nominalQuota=1, borrowingLimit=1, lendingLimit=1
#   - Both in gpu-cohort, reclaimWithinCohort=Any
#
# Timeline:
#   t=0s    Submit team-a-1 and team-a-2 (2 GPUs: 1 own + 1 borrowed)
#   t=30s   Submit team-b-1 (triggers preemption of team-a-2's borrowed GPU)
#           Kueue suspends team-a-2, admits team-b-1
#   t~2m    team-b-1 finishes (90s), GPU frees up
#           Kueue unsuspends team-a-2, new pod starts
#   t~4m    All jobs complete
#
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

header() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }
info()   { printf '\033[0;33m>>> %s\033[0m\n' "$1"; }
status() {
    printf '\n'
    info "Workloads:"
    oc get workloads.kueue.x-k8s.io -A \
        -l demo=kueue-preemption \
        --no-headers 2>/dev/null | column -t || true

    printf '\n'
    info "ClusterQueues:"
    oc get clusterqueues -o custom-columns=\
'NAME:.metadata.name,PENDING:.status.pendingWorkloads,ADMITTED:.status.admittedWorkloads,GPU_USED:.status.flavorsUsage[0].resources[2].total,GPU_BORROWED:.status.flavorsUsage[0].resources[2].borrowed' \
        2>/dev/null || true

    printf '\n'
    info "Jobs:"
    oc get jobs -n first-ai-project -l demo=kueue-preemption \
        -o custom-columns='NAME:.metadata.name,SUSPENDED:.spec.suspend,COMPLETIONS:.status.conditions[0].type,AGE:.metadata.creationTimestamp' \
        --no-headers 2>/dev/null || true
    oc get jobs -n second-ai-project -l demo=kueue-preemption \
        -o custom-columns='NAME:.metadata.name,SUSPENDED:.spec.suspend,COMPLETIONS:.status.conditions[0].type,AGE:.metadata.creationTimestamp' \
        --no-headers 2>/dev/null || true
}

cleanup() {
    header "Cleaning up previous demo jobs"
    oc delete jobs -n first-ai-project -l demo=kueue-preemption --ignore-not-found 2>/dev/null
    oc delete jobs -n second-ai-project -l demo=kueue-preemption --ignore-not-found 2>/dev/null
    sleep 5
}

# ------------------------------------------------------------------
cleanup

header "Step 1: Submit two team-a jobs (will use 1 own + 1 borrowed GPU)"
oc apply -f "$DIR/team-a-job-1.yaml"
oc apply -f "$DIR/team-a-job-2.yaml"

info "Waiting 15s for Kueue to admit both team-a jobs..."
sleep 15
status

header "Step 2: Submit team-b job (triggers reclaimWithinCohort preemption)"
info "Team-b needs its own GPU back — Kueue will suspend team-a-2"
oc apply -f "$DIR/team-b-job-1.yaml"

info "Waiting 15s for preemption to take effect..."
sleep 15
status

header "Step 3: Watching for re-admission of team-a-2"
info "Polling every 20s until team-a-2 is unsuspended and all jobs complete..."
for i in $(seq 1 20); do
    a2_suspended=$(oc get job gpu-training-team-a-2 -n first-ai-project -o jsonpath='{.spec.suspend}' 2>/dev/null || echo "unknown")
    a1_done=$(oc get job gpu-training-team-a-1 -n first-ai-project -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")
    b1_done=$(oc get job gpu-training-team-b-1 -n second-ai-project -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")
    a2_done=$(oc get job gpu-training-team-a-2 -n first-ai-project -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")

    info "Poll $i — team-a-2 suspended=$a2_suspended | completed: a1=$a1_done b1=$b1_done a2=$a2_done"

    if [[ "$a2_done" == "1" && "$a1_done" == "1" && "$b1_done" == "1" ]]; then
        break
    fi
    sleep 20
done

header "Final state"
status

header "Job logs"
info "team-a-1:"
oc logs -n first-ai-project job/gpu-training-team-a-1 --tail=5 2>/dev/null || echo "(no logs yet)"
printf '\n'
info "team-a-2 (preempted then re-admitted):"
oc logs -n first-ai-project job/gpu-training-team-a-2 --all-containers --prefix --tail=10 2>/dev/null || echo "(no logs yet)"
printf '\n'
info "team-b-1:"
oc logs -n second-ai-project job/gpu-training-team-b-1 --tail=5 2>/dev/null || echo "(no logs yet)"

header "Demo complete"
echo "
What happened:
  1. team-a-1 admitted on own GPU quota          (nominalQuota)
  2. team-a-2 admitted on borrowed GPU quota      (borrowingLimit from team-b)
  3. team-b-1 submitted — Kueue reclaimed its GPU (reclaimWithinCohort)
     → team-a-2 was SUSPENDED (not deleted)
  4. When a GPU freed up, Kueue UNSUSPENDED team-a-2
     → Job controller created a new pod
     → Training ran to completion

  Key difference from bare-pod integration:
    Kueue suspends/unsuspends the Job rather than deleting the pod.
    The Job controller handles pod recreation. No Argo retry needed.
"
