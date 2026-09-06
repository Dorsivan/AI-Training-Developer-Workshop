"""
KFP + Kueue CPU Demo Pipeline — Quota Queuing.

Demonstrates Kueue quota-based queuing with KFP using CPU resources.
Three parallel training tasks each request 4 CPU, but the ClusterQueue
only has 6 CPU nominal — so at most one can run at a time.

    preprocess ──┬── train-a (4 CPU) ──┐
                 ├── train-b (4 CPU) ──├── evaluate
                 └── train-c (4 CPU) ──┘

Participants watch Workload objects transition as quota is released:
  - train-a: ADMITTED (4/6 CPU used)
  - train-b: QUEUED   (would need 8, only 6 available)
  - train-c: QUEUED
  - train-a finishes → train-b ADMITTED
  - ...

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


@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311:latest")
def preprocess() -> str:
    import time
    print("Starting preprocessing...")
    time.sleep(30)
    print("Preprocessing complete")
    return "done"


@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311:latest")
def train(model_name: str) -> str:
    import time
    print(f"Training model: {model_name}")
    time.sleep(120)
    print(f"Training complete: {model_name}")
    return f"{model_name}-trained"


@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311:latest")
def evaluate(results: list) -> str:
    import json
    print(f"Evaluating {len(results)} models: {results}")
    print("Evaluation complete")
    return json.dumps({"models": results, "status": "evaluated"})


def use_kueue(task, queue: str = KUEUE_LOCAL_QUEUE, cpu: str = "1", memory: str = "1Gi"):
    kubernetes.add_pod_label(task, "kueue.x-k8s.io/queue-name", queue)
    task.set_cpu_request(cpu)
    task.set_cpu_limit(cpu)
    task.set_memory_request(memory)
    task.set_memory_limit(memory)
    task.set_caching_options(enable_caching=False)
    return task


@dsl.pipeline(
    name="kfp-kueue-cpu-demo",
    description=(
        "KFP pipeline demonstrating Kueue CPU quota queuing. "
        "Three parallel training tasks compete for limited CPU quota."
    ),
)
def cpu_pipeline():
    prep = use_kueue(preprocess(), cpu="500m", memory="256Mi")

    train_a = use_kueue(train(model_name="model-a"), cpu="4", memory="1Gi")
    train_a.after(prep)

    train_b = use_kueue(train(model_name="model-b"), cpu="4", memory="1Gi")
    train_b.after(prep)

    train_c = use_kueue(train(model_name="model-c"), cpu="4", memory="1Gi")
    train_c.after(prep)

    eval_task = use_kueue(
        evaluate(results=[train_a.output, train_b.output, train_c.output]),
        cpu="500m",
        memory="256Mi",
    )


if __name__ == "__main__":
    import pathlib

    queue = KUEUE_LOCAL_QUEUE
    if queue == "team-b-local-queue":
        filename = "kueue_demo_pipeline_team_b.yaml"
    else:
        filename = "kueue_demo_pipeline.yaml"

    output = pathlib.Path(__file__).with_name(filename)
    compiler.Compiler().compile(
        pipeline_func=cpu_pipeline,
        package_path=str(output),
    )
    print(f"Pipeline compiled to {output} (queue: {queue})")
