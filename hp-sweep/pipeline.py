import json
from typing import Dict, List

from kfp import compiler, dsl


PYTHON_IMAGE = "python:3.11-slim"
SKLEARN_PACKAGES = [
    "joblib==1.4.2",
    "pandas==2.2.3",
    "scikit-learn==1.5.2",
]


@dsl.component(
    base_image=PYTHON_IMAGE,
    packages_to_install=SKLEARN_PACKAGES,
)
def create_dataset(
    seed: int,
    development: dsl.Output[dsl.Dataset],
    test: dsl.Output[dsl.Dataset],
):
    """Reserve the final test set before any hyperparameter selection."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    frame = load_breast_cancer(as_frame=True).frame
    development_frame, test_frame = train_test_split(
        frame, test_size=0.20, random_state=seed, stratify=frame["target"]
    )
    for artifact, subset, role in [
        (development, development_frame, "development"),
        (test, test_frame, "held-out-test"),
    ]:
        subset.to_csv(artifact.path, index=False)
        artifact.metadata.update({
            "source": "sklearn.datasets.load_breast_cancer",
            "rows": len(subset), "split_seed": seed, "role": role,
            "row_ids": [int(i) for i in subset.index],
        })


@dsl.component(base_image=PYTHON_IMAGE)
def generate_trials(
    search_space_json: str,
    strategy: str,
    max_trials: int,
    seed: int,
) -> List[Dict]:
    """Expands a JSON search space into a fixed list of trial dictionaries."""
    import json
    import math
    import random
    import sys

    search_space = json.loads(search_space_json)
    required = {"n_estimators", "max_depth", "min_samples_split"}
    if not isinstance(search_space, dict) or set(search_space) != required:
        raise ValueError(f"Expected exactly these search keys: {sorted(required)}")
    names = sorted(search_space)
    values = []
    for name in names:
        candidates = search_space[name]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{name} must be a non-empty list")
        lower = 2 if name == "min_samples_split" else 1
        if any(type(v) is not int or v < lower for v in candidates):
            raise ValueError(f"{name} candidates must be integers >= {lower}")
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"{name} contains duplicate candidates")
        values.append(candidates)

    total = math.prod(len(v) for v in values)
    output_cap = 512
    if strategy == "grid":
        if total > output_cap:
            raise ValueError(f"Grid has {total} trials; limit is {output_cap}")
        indices = range(total)
    elif strategy == "random":
        if max_trials <= 0 or max_trials > output_cap:
            raise ValueError(f"max_trials must be in 1..{output_cap}")
        if total > sys.maxsize:
            raise ValueError("Search space exceeds supported index range")
        indices = random.Random(seed).sample(range(total), min(max_trials, total))
    else:
        raise ValueError("strategy must be 'grid' or 'random'")

    selected = []
    for index in indices:
        trial = {}
        for name, candidates in reversed(list(zip(names, values))):
            index, offset = divmod(index, len(candidates))
            trial[name] = candidates[offset]
        selected.append(trial)
    if len(json.dumps(selected).encode("utf-8")) > 128 * 1024:
        raise ValueError("Trial list exceeds this guide's 128 KiB output budget")
    print(f"Generated {len(selected)} of {total} possible trial configurations")
    return selected


@dsl.component(
    base_image=PYTHON_IMAGE,
    packages_to_install=SKLEARN_PACKAGES,
)
def train_trial(
    dataset: dsl.Input[dsl.Dataset],
    trial: Dict,
    seed: int,
) -> str:
    """Trains one trial and returns a compact JSON result parameter."""
    import hashlib
    import json
    import math

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv(dataset.path)
    features = frame.drop(columns=["target"])
    target = frame["target"]

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=seed,
        stratify=target,
    )

    parameters = {
        "n_estimators": int(trial["n_estimators"]),
        "max_depth": int(trial["max_depth"]),
        "min_samples_split": int(trial["min_samples_split"]),
    }

    model = RandomForestClassifier(
        **parameters,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    metric_value = float(accuracy_score(y_valid, predictions))

    if not math.isfinite(metric_value):
        raise ValueError(f"Trial produced a non-finite metric: {metric_value}")

    canonical_parameters = json.dumps(parameters, sort_keys=True)
    trial_id = hashlib.sha256((canonical_parameters + ":" + str(seed)).encode("utf-8")).hexdigest()[:12]

    result = {
        "trial_id": trial_id,
        "metric_name": "accuracy",
        "metric_value": metric_value,
        "parameters": parameters,
        "seed": seed,
    }

    result_json = json.dumps(result, sort_keys=True)
    print(result_json)
    return result_json


@dsl.component(base_image=PYTHON_IMAGE)
def select_best(
    trial_results_json: List[str],
    direction: str,
) -> str:
    """Selects the best valid result returned by the parallel trials."""
    import json
    import math

    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")

    valid_results = []
    for raw_result in trial_results_json:
        result = json.loads(raw_result)
        metric = float(result["metric_value"])
        if not math.isfinite(metric):
            raise ValueError("Trial metric must be finite")
        if result.get("metric_name") != "accuracy":
            raise ValueError("All trials must report accuracy")
        if not isinstance(result.get("parameters"), dict) or not result.get("trial_id"):
            raise ValueError("Missing trial parameters or ID")
        result["metric_value"] = metric
        valid_results.append(result)

    if not valid_results:
        raise RuntimeError("No successful trial produced a finite metric")

    best = (
        max(valid_results, key=lambda item: (item["metric_value"], item["trial_id"]))
        if direction == "maximize"
        else min(valid_results, key=lambda item: (item["metric_value"], item["trial_id"]))
    )

    best["completed_trial_count"] = len(valid_results)
    best_json = json.dumps(best, sort_keys=True)
    print(f"Best trial: {best_json}")
    return best_json


@dsl.component(
    base_image=PYTHON_IMAGE,
    packages_to_install=SKLEARN_PACKAGES,
)
def train_final_model(
    dataset: dsl.Input[dsl.Dataset],
    test_dataset: dsl.Input[dsl.Dataset],
    best_result_json: str,
    seed: int,
    model: dsl.Output[dsl.Model],
    report: dsl.Output[dsl.Artifact],
) -> float:
    """Retrains a final model with the selected hyperparameters."""
    import json

    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    best_result = json.loads(best_result_json)
    parameters = best_result["parameters"]

    frame = pd.read_csv(dataset.path)
    features = frame.drop(columns=["target"])
    target = frame["target"]

    test_frame = pd.read_csv(test_dataset.path)
    x_train, y_train = features, target
    x_test = test_frame.drop(columns=["target"])
    y_test = test_frame["target"]

    final_model = RandomForestClassifier(
        n_estimators=int(parameters["n_estimators"]),
        max_depth=int(parameters["max_depth"]),
        min_samples_split=int(parameters["min_samples_split"]),
        random_state=seed,
        n_jobs=1,
    )
    final_model.fit(x_train, y_train)

    test_accuracy = float(accuracy_score(y_test, final_model.predict(x_test)))
    joblib.dump(final_model, model.path)

    final_report = {
        "selected_trial": best_result,
        "test_accuracy": test_accuracy,
    }
    with open(report.path, "w", encoding="utf-8") as report_file:
        json.dump(final_report, report_file, indent=2, sort_keys=True)

    model.metadata["framework"] = "scikit-learn"
    model.metadata["algorithm"] = "RandomForestClassifier"
    model.metadata["test_accuracy"] = test_accuracy
    model.metadata["hyperparameters"] = parameters
    report.metadata["test_accuracy"] = test_accuracy

    print(json.dumps(final_report, indent=2, sort_keys=True))
    return test_accuracy


DEFAULT_SEARCH_SPACE = json.dumps(
    {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 8, 16],
        "min_samples_split": [2, 5],
    },
    sort_keys=True,
)


@dsl.pipeline(name="random-forest-hyperparameter-sweep")
def hyperparameter_sweep_pipeline(
    search_space_json: str = DEFAULT_SEARCH_SPACE,
    strategy: str = "grid",
    max_trials: int = 12,
    direction: str = "maximize",
    seed: int = 42,
):
    dataset_task = create_dataset(seed=seed)

    trials_task = generate_trials(
        search_space_json=search_space_json,
        strategy=strategy,
        max_trials=max_trials,
        seed=seed,
    )

    with dsl.ParallelFor(
        items=trials_task.output,
        parallelism=4,
        name="hyperparameter-trials",
    ) as trial:
        trial_task = train_trial(
            dataset=dataset_task.outputs["development"],
            trial=trial,
            seed=seed,
        )
        trial_task.set_cpu_request("1")
        trial_task.set_memory_request("1Gi")

    best_task = select_best(
        trial_results_json=dsl.Collected(trial_task.output),
        direction=direction,
    )

    final_task = train_final_model(
        dataset=dataset_task.outputs["development"],
        test_dataset=dataset_task.outputs["test"],
        best_result_json=best_task.output,
        seed=seed,
    )
    final_task.set_cpu_request("2")
    final_task.set_memory_request("2Gi")


if __name__ == "__main__":
    import pathlib

    output = pathlib.Path(__file__).with_name("hyperparameter_sweep_pipeline.yaml")
    compiler.Compiler().compile(
        pipeline_func=hyperparameter_sweep_pipeline,
        package_path=str(output),
    )
    print(f"Pipeline compiled to {output}")
