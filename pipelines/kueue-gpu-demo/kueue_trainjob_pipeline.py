"""
KFP + Kueue GPU Demo — PyTorchJob Pattern.

The GPU training step launches a Kueue-managed PyTorchJob instead of
requesting GPUs directly on the Argo pod.  This gives Kueue native
suspend/unsuspend lifecycle over the training workload:

  KFP component (lightweight, no GPU)
       │
       │  creates + watches
       ▼
  PyTorchJob (kubeflow.org/v1)  ←  Kueue manages this
       │
       ▼
  GPU pod (suspend / preempt / re-admit)
       │
       ▼
  PyTorchJob completes  →  KFP component succeeds  →  next step

Kueue preemption suspends the PyTorchJob (not the Argo pod), so the
KFP pipeline node stays Running throughout.  No Argo retry needed.

Demo scenario (2 physical GPUs):
  1. Submit 2 runs to team-a  →  1 own + 1 borrowed GPU
  2. Submit 1 run to team-b  →  Kueue preempts team-a's borrowed
     PyTorchJob via reclaimWithinCohort, then re-admits when GPU frees up

Requires:
  - Kubeflow Training Operator (PyTorchJob CRD)
  - Kueue with PyTorchJob integration enabled

Compile:
    python kueue_trainjob_pipeline.py
    KUEUE_LOCAL_QUEUE=team-b-local-queue python kueue_trainjob_pipeline.py

Submit:
    python submit.py team-a --trainjob
    python submit.py team-b --trainjob
"""

import os

from kfp import compiler, dsl
from kfp import kubernetes

KUEUE_LOCAL_QUEUE = os.environ.get("KUEUE_LOCAL_QUEUE", "team-a-local-queue")


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def preprocessing() -> str:
    import time
    print("Preprocessing data...")
    time.sleep(20)
    print("Preprocessing complete")
    return "preprocessed"


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["kubernetes"],
)
def run_training_job(
    data_status: str,
    kueue_queue: str,
    train_duration: int = 300,
    num_workers: int = 0,
    gpus_per_replica: int = 1,
    poll_interval: int = 10,
) -> str:
    """Launch a Kueue-managed PyTorchJob and wait for it to complete.

    This component does NOT request GPUs itself.  It creates a
    kubeflow.org/v1 PyTorchJob with Kueue labels, then polls
    until completion.  Kueue can freely suspend/preempt/re-admit
    the PyTorchJob without affecting this Argo node.
    """
    import datetime
    import time
    import uuid

    from kubernetes import client, config

    GROUP = "kubeflow.org"
    VERSION = "v1"
    PLURAL = "pytorchjobs"

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()
    core_v1 = client.CoreV1Api()

    namespace = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read()
    job_name = f"gpu-train-{uuid.uuid4().hex[:8]}"

    training_script = f"""
import time, datetime, subprocess, os
print(f"[{{datetime.datetime.now():%H:%M:%S}}] Training started (input: {data_status})")
subprocess.run(["nvidia-smi"], check=False)
duration = {train_duration}
for elapsed in range(15, duration + 1, 15):
    time.sleep(15)
    pct = 100 * elapsed // duration
    print(f"[{{datetime.datetime.now():%H:%M:%S}}] Progress: {{elapsed}}/{{duration}}s ({{pct}}%)")
print(f"[{{datetime.datetime.now():%H:%M:%S}}] Training complete")
"""

    container = {
        "name": "pytorch",
        "image": "quay.io/modh/cuda-notebooks:cuda-jupyter-minimal-ubi9-python-3.11-20250630",
        "command": ["python3", "-c", training_script],
        "resources": {
            "requests": {
                "cpu": "2",
                "memory": "4Gi",
                "nvidia.com/gpu": str(gpus_per_replica),
            },
            "limits": {
                "cpu": "2",
                "memory": "4Gi",
                "nvidia.com/gpu": str(gpus_per_replica),
            },
        },
    }

    replica_spec = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "template": {
            "spec": {
                "containers": [container],
            },
        },
    }

    pytorch_job = {
        "apiVersion": f"{GROUP}/{VERSION}",
        "kind": "PyTorchJob",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "kueue.x-k8s.io/queue-name": kueue_queue,
                "kfp-training-job": "true",
            },
        },
        "spec": {
            "suspend": True,
            "pytorchReplicaSpecs": {
                "Master": replica_spec,
            },
        },
    }

    if num_workers > 0:
        worker_spec = {
            "replicas": num_workers,
            "restartPolicy": "OnFailure",
            "template": {
                "spec": {
                    "containers": [container],
                },
            },
        }
        pytorch_job["spec"]["pytorchReplicaSpecs"]["Worker"] = worker_spec

    total_gpus = (1 + num_workers) * gpus_per_replica
    print(f"[{datetime.datetime.now():%H:%M:%S}] Creating PyTorchJob {job_name} "
          f"(queue={kueue_queue}, master=1, workers={num_workers}, "
          f"gpus/replica={gpus_per_replica}, total_gpus={total_gpus}, "
          f"duration={train_duration}s)")
    custom_api.create_namespaced_custom_object(
        group=GROUP, version=VERSION, namespace=namespace,
        plural=PLURAL, body=pytorch_job,
    )

    while True:
        pj = custom_api.get_namespaced_custom_object(
            group=GROUP, version=VERSION, namespace=namespace,
            plural=PLURAL, name=job_name,
        )

        conditions = {
            c["type"]: c["status"]
            for c in pj.get("status", {}).get("conditions", [])
        }
        suspended = pj.get("spec", {}).get("suspend", False)

        if conditions.get("Succeeded") == "True":
            print(f"[{datetime.datetime.now():%H:%M:%S}] PyTorchJob {job_name} succeeded")
            break

        if conditions.get("Failed") == "True":
            msg = next(
                (c.get("message", "") for c in pj["status"]["conditions"]
                 if c["type"] == "Failed"),
                "",
            )
            raise RuntimeError(f"PyTorchJob {job_name} failed: {msg}")

        if suspended:
            status = "suspended (Kueue preempted or waiting for admission)"
        elif conditions.get("Running") == "True":
            status = "running"
        elif conditions.get("Created") == "True":
            status = "created, waiting for pods"
        else:
            status = "pending"

        print(f"[{datetime.datetime.now():%H:%M:%S}] PyTorchJob {job_name}: {status}")
        time.sleep(poll_interval)

    try:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"training.kubeflow.org/job-name={job_name}",
        )
        for pod in pods.items:
            try:
                log = core_v1.read_namespaced_pod_log(
                    name=pod.metadata.name, namespace=namespace, tail_lines=5,
                )
                print(f"--- {pod.metadata.name} (last 5 lines) ---")
                print(log)
            except Exception:
                pass
    except Exception:
        pass

    return "trained"


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
)
def evaluation(model_status: str) -> str:
    import time
    print(f"Evaluating model (input: {model_status})")
    time.sleep(20)
    print("Evaluation complete")
    return "evaluated"


@dsl.pipeline(
    name="kfp-kueue-gpu-demo",
    description=(
        "GPU pipeline with Kueue-managed PyTorchJob. "
        "The KFP component launches and watches a kubeflow.org/v1 "
        "PyTorchJob that Kueue can suspend/preempt/re-admit natively."
    ),
)
def gpu_pipeline():
    prep = preprocessing()
    prep.set_cpu_request("500m").set_cpu_limit("500m")
    prep.set_memory_request("256Mi").set_memory_limit("256Mi")
    prep.set_caching_options(enable_caching=False)

    train = run_training_job(
        data_status=prep.output,
        kueue_queue=KUEUE_LOCAL_QUEUE,
        train_duration=300,
    )
    train.set_cpu_request("500m").set_cpu_limit("500m")
    train.set_memory_request("256Mi").set_memory_limit("256Mi")
    train.set_caching_options(enable_caching=False)

    eval_task = evaluation(model_status=train.output)
    eval_task.set_cpu_request("500m").set_cpu_limit("500m")
    eval_task.set_memory_request("256Mi").set_memory_limit("256Mi")
    eval_task.set_caching_options(enable_caching=False)


if __name__ == "__main__":
    import pathlib

    queue = KUEUE_LOCAL_QUEUE
    if queue == "team-b-local-queue":
        filename = "kueue_trainjob_pipeline_team_b.yaml"
    else:
        filename = "kueue_trainjob_pipeline.yaml"

    output = pathlib.Path(__file__).with_name(filename)
    compiler.Compiler().compile(
        pipeline_func=gpu_pipeline,
        package_path=str(output),
    )
    print(f"Pipeline compiled to {output} (queue: {queue})")
