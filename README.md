# OpenShift AI Pipelines Example

Example ML pipelines for **Red Hat OpenShift AI** (RHOAI 3.4+), with Kueue
scheduling, Vault secret retrieval, and data-quality validation.

## Repository structure

```
pipelines/
  iris-training/          Iris classifier training pipeline
  data-validation/        Data validation & conditional processing pipeline
infrastructure/
  kueue/                  Kueue resource-quota & preemption manifests
  vault/                  HashiCorp Vault deployment manifests
  nfs-server/             NFS Ganesha server for shared storage
  lb-test/                LoadBalancer smoke-test pod + service
docs/                     Reference documentation (PDFs, etc.)
requirements.txt          Python dependencies for pipeline compilation
```

---

## Pipelines

### Iris Classifier Training (`pipelines/iris-training/`)

Trains a scikit-learn RandomForest classifier on the Iris dataset, evaluates it,
and conditionally exports the model when accuracy meets a configurable threshold.

```
fetch_secret_from_vault   fetch_data_from_uri
              \                /
           preprocess_data
                  |
             train_model
                  |
            evaluate_model
                  |
           [accuracy >= 0.90?]
                  |
            export_model
```

| File | Description |
|------|-------------|
| `pipeline.py` | KFP v2 Python pipeline source |
| `iris_training_pipeline.yaml` | Compiled pipeline YAML (ready to upload) |
| `uri-connection.yaml` | OpenShift AI data connection for the Iris CSV |
| `vault-uri-connection.yaml` | OpenShift AI data connection for Vault endpoint |

#### Pipeline parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vault_uri` | *(Vault KV endpoint)* | URI for fetching the API key |
| `vault_token` | *(empty)* | Vault authentication token |
| `data_uri` | *(GitHub raw CSV)* | URI of the Iris CSV dataset |
| `test_size` | `0.2` | Fraction of data held out for testing |
| `n_estimators` | `100` | Number of trees in the RandomForest |
| `max_depth` | `5` | Maximum tree depth |
| `accuracy_threshold` | `0.90` | Minimum accuracy to trigger model export |

### Data Validation & Processing (`pipelines/data-validation/`)

Validates ingested data quality and conditionally transforms good data or
quarantines bad data with an alert notification.

```
ingest_data
     |
validate_data
     |
[quality >= 0.8?]
   /         \
transform   quarantine
   |            |
report      send_alert
```

| File | Description |
|------|-------------|
| `data_validation_pipeline.py` | KFP v2 Python pipeline source |
| `data_validation_pipeline.yaml` | Compiled pipeline YAML (ready to upload) |
| `pipeline-pvc.yaml` | PVC manifest for shared data between steps |

#### Pipeline parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quality_threshold` | `0.8` | Minimum quality score to take the good-data path |
| `num_rows` | `100` | Number of synthetic rows to generate |
| `null_fraction` | `0.05` | Fraction of cells to randomly null out |
| `pvc_size` | `1Gi` | Size of the dynamically created PVC |
| `storage_class` | *(default)* | Storage class for the PVC |

---

## Quick start: upload a pipeline to OpenShift AI

You can use the **pre-compiled YAML** files directly without installing anything
locally.

### 1. Create a project (if you don't have one)

1. Log in to the **OpenShift AI dashboard**.
2. Click **Projects** in the left sidebar.
3. Click **Create project**, give it a name (e.g. `iris-pipeline-demo`), and
   click **Create**.

### 2. Configure a pipeline server

Before importing pipelines you need a pipeline server in your project:

1. Open your project and go to the **Pipelines** tab.
2. Click **Configure pipeline server**.
3. Provide an S3-compatible object storage connection (e.g. MinIO or AWS S3)
   for pipeline artifact storage:
   - **Access key**, **Secret key**, **Endpoint**, **Bucket name**
4. Click **Configure**.
5. Wait until the pipeline server status shows **Available**.

### 3. Import the pipeline

1. In your project, go to the **Pipelines** tab.
2. Click **Import pipeline**.
3. Enter a name (e.g. `Iris Classifier Training`).
4. Choose **Upload a file** and select the compiled YAML from the relevant
   `pipelines/` subdirectory.
5. Click **Import**.

### 4. Create a run

1. On the **Pipelines** tab, find your imported pipeline.
2. Click the **action menu** (three dots) and select **Create run**.
3. Adjust pipeline parameters if desired (or keep defaults).
4. Click **Create** to start the run.

### 5. Monitor the run

1. Go to the **Runs** tab in your project.
2. Click on your run to see the pipeline graph and step status.
3. Click individual steps to view logs and output artifacts.

---

## Recompile from source (optional)

If you modify a pipeline `.py` file, recompile the YAML:

```bash
pip install -r requirements.txt
python pipelines/iris-training/pipeline.py
python pipelines/data-validation/data_validation_pipeline.py
```

> **Note:** Requires Python 3.9-3.12 (kfp 2.x does not yet support Python 3.13+).
> If using Fedora 43+ with Python 3.14, compile in a container:
> ```bash
> podman run --rm -v "$(pwd):/work:Z" -w /work python:3.11-slim \
>   bash -c "pip install -q kfp==2.12.1 && python pipelines/iris-training/pipeline.py"
> ```

---

## Infrastructure

### Kueue (`infrastructure/kueue/`)

Both pipelines are configured for [Kueue](https://kueue.sigs.k8s.io/) scheduling.
Kueue manages resource quotas and enables **preemption** — higher-priority pipeline
runs can evict lower-priority ones when the cluster is at capacity.

Apply the manifests **once per cluster** (requires cluster-admin):

```bash
oc apply -f infrastructure/kueue/00-resource-flavor.yaml
oc apply -f infrastructure/kueue/01-workload-priority-classes.yaml
oc apply -f infrastructure/kueue/02-cluster-queue.yaml
oc apply -f infrastructure/kueue/04-priority-classes.yaml

# Edit 03-local-queue.yaml to set the correct namespace first
oc apply -f infrastructure/kueue/03-local-queue.yaml

# Requires Kyverno
oc apply -f infrastructure/kueue/05-kyverno-priority-class-policy.yaml
```

| Resource | Purpose |
|----------|---------|
| **ResourceFlavor** (`default-flavor`) | CPU/memory workloads — any node |
| **ResourceFlavor** (`gpu-flavor`) | GPU workloads — node affinity via `nvidia.com/gpu.present: "true"` |
| **WorkloadPriorityClass** | Three Kueue tiers: `pipeline-low-priority` (100), `pipeline-default-priority` (1000), `pipeline-high-priority` (10000) |
| **ClusterQueue** | Enforces resource quotas (8 CPU / 16 Gi / 2 GPU) with `withinClusterQueue: LowerPriority` preemption |
| **LocalQueue** | Namespaced queue that feeds into the ClusterQueue |
| **PriorityClass** | Three Kubernetes scheduler tiers matching the Kueue tiers above |
| **Kyverno ClusterPolicy** | Mutates pods to copy the `kueue.x-k8s.io/priority-class` label into `spec.priorityClassName` |

#### How preemption works

Preemption operates at **two layers**:

1. **Kueue admission** — the `kueue.x-k8s.io/priority-class` label references a
   WorkloadPriorityClass, controlling which workloads Kueue admits or evicts.
2. **Kubernetes scheduler** — the Kyverno policy copies that same label value into
   `spec.priorityClassName`, so the scheduler also preempts lower-priority pods.

To run a pipeline at a different priority, change the `KUEUE_PRIORITY` constant
at the top of the pipeline file and recompile:

```python
KUEUE_PRIORITY = "pipeline-high-priority"  # or "pipeline-low-priority"
```

#### Enabling pod integration

Kueue must be configured to manage plain pods (KFP tasks run as individual pods).
If you installed Kueue via the operator, ensure the pod integration framework is
enabled in the Kueue configuration:

```yaml
integrations:
  frameworks:
    - "pod"
  podOptions:
    namespaceSelector:
      matchExpressions:
        - key: kubernetes.io/metadata.name
          operator: NotIn
          values: [kube-system, kueue-system]
```

### Vault (`infrastructure/vault/`)

HashiCorp Vault deployment for dev/demo secret management. See
`infrastructure/vault/CREDENTIALS.md` for access details.

### NFS Server (`infrastructure/nfs-server/`)

NFS Ganesha server exposed via LoadBalancer for shared storage:

```bash
oc apply -f infrastructure/nfs-server/nfs-server.yaml
bash infrastructure/nfs-server/post-apply-scc.sh
```

### LoadBalancer Test (`infrastructure/lb-test/`)

Simple nginx pod + LoadBalancer service for verifying LB connectivity.

---

## Base images

All pipeline steps use `registry.redhat.io/ubi9/python-311:latest` (Red Hat
Universal Base Image). If your cluster cannot pull from `registry.redhat.io`,
swap the `base_image` in the pipeline `.py` files to a mirror or to
`quay.io/modh/runtime-images:runtime-python-3.11-ubi9`.
