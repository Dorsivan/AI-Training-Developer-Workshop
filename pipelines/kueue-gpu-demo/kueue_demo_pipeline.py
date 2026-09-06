"""
KFP + Kueue GPU Demo Pipeline — Preemption & Borrowing.

Demonstrates Kueue GPU quota management and cohort preemption with KFP.
Each pipeline run requests 1 GPU for the training stage.  Two
ClusterQueues share a cohort, each with nominalQuota=1 and
borrowingLimit=1 GPU.

Demo scenario (2 physical GPUs):
  1. Submit 2 runs to team-a  →  uses 1 own + 1 borrowed GPU
  2. Submit 1 run to team-b  →  Kueue preempts team-a's borrowed GPU
     via reclaimWithinCohort, then admits team-b's training pod

    preprocess (CPU) → training (GPU, 5 min hold) → evaluation (CPU)

Compile for team-a (default):
    python kueue_demo_pipeline.py

Compile for team-b:
    KUEUE_LOCAL_QUEUE=team-b-local-queue python kueue_demo_pipeline.py

Submit:
    python submit.py team-a
    python submit.py team-b
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
    base_image="quay.io/modh/cuda-notebooks:cuda-jupyter-minimal-ubi9-python-3.11-20250630",
)
def gpu_training(data_status: str) -> str:
    import subprocess
    import time

    print(f"GPU training started (input: {data_status})")
    subprocess.run(["nvidia-smi"], check=False)

    print("Holding GPU for 5 minutes...")
    time.sleep(300)

    print("GPU training finished")
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


def use_kueue(task, queue: str = KUEUE_LOCAL_QUEUE, cpu: str = "1", memory: str = "1Gi"):
    kubernetes.add_pod_label(task, "kueue.x-k8s.io/queue-name", queue)
    task.set_cpu_request(cpu)
    task.set_cpu_limit(cpu)
    task.set_memory_request(memory)
    task.set_memory_limit(memory)
    task.set_caching_options(enable_caching=False)
    return task


@dsl.pipeline(
    name="kfp-kueue-gpu-demo",
    description=(
        "GPU pipeline whose training Pod is admitted through Kueue. "
        "Kueue delays only the GPU-constrained stage, not the whole pipeline."
    ),
)
def gpu_pipeline():
    prep = use_kueue(preprocessing(), cpu="500m", memory="256Mi")

    train = use_kueue(gpu_training(data_status=prep.output), cpu="2", memory="4Gi")
    train.set_accelerator_type("nvidia.com/gpu")
    train.set_accelerator_limit(1)
    train.set_retry(
        num_retries=3,
        backoff_duration="30s",
        backoff_factor=2.0,
        backoff_max_duration="5m",
    )

    eval_task = use_kueue(evaluation(model_status=train.output), cpu="500m", memory="256Mi")


if __name__ == "__main__":
    import pathlib

    queue = KUEUE_LOCAL_QUEUE
    if queue == "team-b-local-queue":
        filename = "kueue_demo_pipeline_team_b.yaml"
    else:
        filename = "kueue_demo_pipeline.yaml"

    output = pathlib.Path(__file__).with_name(filename)
    compiler.Compiler().compile(
        pipeline_func=gpu_pipeline,
        package_path=str(output),
    )
    print(f"Pipeline compiled to {output} (queue: {queue})")
