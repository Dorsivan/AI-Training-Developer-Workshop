import json

from client_config import make_client

client = make_client()

run = client.create_run_from_pipeline_package(
    "hyperparameter_sweep_pipeline.yaml",
    arguments={
        "search_space_json": json.dumps(
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [4, 8, 16],
                "min_samples_split": [2, 5],
            }
        ),
        "strategy": "random",
        "max_trials": 8,
        "direction": "maximize",
        "seed": 42,
    },
    experiment_name="hyperparameter-sweeps",
    run_name="random-forest-random-sweep",
)

print(run.run_id)
