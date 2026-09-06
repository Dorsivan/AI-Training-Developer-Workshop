# KFP + Kueue GPU Demo — TrainJob Pattern

Demonstrates Kueue GPU quota management and preemption with KFP pipelines on OpenShift AI, using the TrainJob pattern for clean suspend/re-admit lifecycle.

## The Problem

When KFP runs a GPU workload directly on an Argo pod, Kueue can only preempt by **deleting** the pod — which kills the entire pipeline. There is no way for Kueue to gracefully suspend and resume a bare pod.

## The Solution: TrainJob Pattern

Instead of requesting GPUs on the Argo pod, the KFP component launches a `TrainJob` (`trainer.kubeflow.org/v1alpha1`) and polls it until completion. This creates two nested control loops:

```
KFP pipeline (Argo Workflow)
  └─ preprocessing (CPU only)
  └─ run_training_job (CPU only, lightweight poller)
  │     └─ creates TrainJob ← Kueue manages this
  │     │     └─ GPU pod (suspend / preempt / re-admit)
  │     └─ polls until TrainJob completes
  └─ evaluation (CPU only)
```

- **KFP** manages the pipeline DAG (preprocessing → training → evaluation)
- **Kueue** manages the GPU workload lifecycle (admit / suspend / re-admit the TrainJob)
- When Kueue preempts, it **suspends the TrainJob** — the KFP poller stays running and logs `suspended (Kueue preempted or waiting for admission)` until the TrainJob is re-admitted
- No Argo retry needed — the pipeline never fails

## Prerequisites

### 1. Kubeflow Training Operator v2 (TrainJob CRD)

Install the trainer controller and ClusterTrainingRuntimes:

```bash
oc apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=v2.3.0"
oc apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=v2.3.0"
```

Verify:

```bash
oc get crd trainjobs.trainer.kubeflow.org
oc get clustertrainingruntimes
oc get pods -n kubeflow-system
```

### 2. Kueue with TrainJob Framework Integration

The Kueue config must include `trainer.kubeflow.org/trainjob` in its frameworks list. Check:

```bash
oc get configmap -n openshift-kueue-operator kueue-manager-config -o yaml | grep -A10 'frameworks:'
```

If missing, patch the ConfigMap to add it and restart the controller:

```bash
# Edit the configmap to add trainer.kubeflow.org/trainjob under integrations.frameworks
oc edit configmap -n openshift-kueue-operator kueue-manager-config
oc rollout restart deploy -n openshift-kueue-operator kueue-controller-manager
```

### 3. Kueue Queues (ClusterQueues + LocalQueues)

The demo expects two ClusterQueues in a shared cohort with borrowing and preemption enabled:

| Resource | team-a | team-b |
|---|---|---|
| ClusterQueue | `team-a-cluster-queue` | `team-b-cluster-queue` |
| LocalQueue | `team-a-local-queue` (in `first-ai-project`) | `team-b-local-queue` (in `second-ai-project`) |
| Cohort | `gpu-cohort` | `gpu-cohort` |
| nominalQuota (GPU) | 1 | 1 |
| borrowingLimit (GPU) | 1 | 1 |
| lendingLimit (GPU) | 1 | 1 |
| preemption.reclaimWithinCohort | Any | Any |

### 4. RBAC for the KFP Service Account

The KFP pipeline runner service account needs permission to create/watch TrainJobs and read pod logs:

```bash
oc apply -f rbac-trainjob.yaml -n first-ai-project
oc apply -f rbac-trainjob.yaml -n second-ai-project
```

This grants the `pipeline-runner-dspa` service account access to `trainjobs` in `trainer.kubeflow.org`.

### 5. Namespace Labels

Namespaces must be labeled for Kueue management:

```bash
oc label namespace first-ai-project kueue.openshift.io/managed=true
oc label namespace second-ai-project kueue.openshift.io/managed=true
```

## Key Design Decisions in the Pipeline Code

### The KFP component requests no GPUs

The `run_training_job` component runs on a lightweight pod (500m CPU, 256Mi memory). GPUs are only requested inside the TrainJob spec. This means Kueue never touches the Argo pod.

### TrainJob is created with `suspend: True`

The TrainJob starts suspended so Kueue controls when it gets admitted:

```python
"spec": {
    "suspend": True,
    "runtimeRef": {"name": "torch-distributed"},
    "trainer": {
        "image": "...",
        "command": ["python3", "-c", training_script],
        "numNodes": 1,
        "resourcesPerNode": {
            "requests": {"nvidia.com/gpu": "1"},
            "limits": {"nvidia.com/gpu": "1"},
        },
    },
}
```

### `runtimeRef` is required

TrainJob v2 requires a `runtimeRef` pointing to a `ClusterTrainingRuntime`. The demo uses `torch-distributed` which is installed with the runtimes overlay.

### The Kueue queue label goes on the TrainJob, not the Argo pod

```python
"labels": {
    "kueue.x-k8s.io/queue-name": kueue_queue,
}
```

### Polling checks for both `Complete` and `Succeeded` conditions

Different TrainJob API versions use different condition types:

```python
if conditions.get("Succeeded") == "True" or conditions.get("Complete") == "True":
    break
```

## Running the Demo

### Compile

```bash
python kueue_trainjob_pipeline.py
KUEUE_LOCAL_QUEUE=team-b-local-queue python kueue_trainjob_pipeline.py
```

### Submit

```bash
# Submit 2 team-a runs (uses 1 own + 1 borrowed GPU)
python submit.py team-a --trainjob
python submit.py team-a --trainjob

# Wait ~40 seconds for training to start, then submit team-b
sleep 40
python submit.py team-b --trainjob
```

### What to Expect

1. Both team-a TrainJobs get admitted (1 on own quota, 1 borrowing from team-b)
2. Team-b's TrainJob triggers `reclaimWithinCohort` preemption
3. Kueue suspends team-a's borrowed TrainJob — the KFP workflow stays Running
4. Team-a #1 and team-b run in parallel (2 GPUs)
5. When team-a #1 completes, Kueue re-admits team-a #2
6. All 3 KFP pipelines complete successfully

### Monitoring

```bash
# Watch TrainJobs
oc get trainjobs -A -w

# Watch Kueue workloads
oc get workloads -A -w

# Check ClusterQueue GPU usage
oc get clusterqueues

# Check KFP workflow status
oc get workflows -n first-ai-project
oc get workflows -n second-ai-project
```

## Files

| File | Description |
|---|---|
| `kueue_trainjob_pipeline.py` | Pipeline code — TrainJob pattern |
| `submit.py` | Submission script (`--trainjob` flag selects this pattern) |
| `rbac-trainjob.yaml` | Role + RoleBinding for TrainJob CRUD |
| `kueue_demo_pipeline.py` | Alternative: direct pod integration with Argo retry (fallback approach) |
| `jobs/` | Standalone batch/v1 Job manifests for native Kueue demo (no KFP) |
