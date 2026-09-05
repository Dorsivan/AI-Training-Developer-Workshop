"""
Example AI Pipeline for Red Hat OpenShift AI Pipelines.

This pipeline demonstrates a typical ML workflow with external data connections:
  1. Fetch an API key from a Vault URI data connection
  2. Fetch dataset from a second URI data connection
  3. Preprocess and split the data
  4. Train a scikit-learn RandomForest model
  5. Evaluate the model
  6. Conditionally export the model if accuracy meets threshold

Compile with:
    python pipeline.py

Then upload the generated YAML to the OpenShift AI dashboard.
"""

from kfp import dsl, compiler
from kfp import kubernetes

# Kueue scheduling configuration (compile-time values — recompile to change).
KUEUE_LOCAL_QUEUE = "pipeline-local-queue"
KUEUE_PRIORITY = "pipeline-default-priority"


def apply_kueue_config(task, queue_name=KUEUE_LOCAL_QUEUE, priority_class=KUEUE_PRIORITY):
    """Configure a pipeline task for Kueue scheduling and preemption."""
    kubernetes.add_pod_label(task, "kueue.x-k8s.io/queue-name", queue_name)
    kubernetes.add_pod_label(task, "kueue.x-k8s.io/priority-class", priority_class)


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
)
def fetch_secret_from_vault(
    vault_uri: str,
    vault_token: str,
    secret_key: str = "key",
) -> str:
    """Fetch a secret value from HashiCorp Vault via its KV v2 API."""
    import json
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(vault_uri)
    req.add_header("X-Vault-Token", vault_token)

    with urllib.request.urlopen(req, context=ctx) as resp:
        body = json.loads(resp.read())

    value = body["data"]["data"][secret_key]
    masked = value[:4] + "****" + value[-4:]
    print(f"Retrieved secret '{secret_key}' from Vault: {masked}")
    return value


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
)
def fetch_data_from_uri(
    data_uri: str,
    raw_data_path: dsl.OutputPath("csv"),
):
    """Download a CSV dataset from a URI (HTTP/HTTPS)."""
    import urllib.request

    print(f"Fetching data from: {data_uri}")
    urllib.request.urlretrieve(data_uri, raw_data_path)

    with open(raw_data_path) as f:
        line_count = sum(1 for _ in f) - 1
    print(f"Downloaded {line_count} rows")


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
    packages_to_install=["scikit-learn==1.5.2", "pandas==2.2.3"],
)
def preprocess_data(
    raw_data_path: dsl.InputPath("csv"),
    train_features_path: dsl.OutputPath("csv"),
    test_features_path: dsl.OutputPath("csv"),
    train_labels_path: dsl.OutputPath("csv"),
    test_labels_path: dsl.OutputPath("csv"),
    label_column: str = "species",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Preprocess a raw CSV: encode labels, scale features, split train/test."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df[label_column]), name="target")
    X = df.drop(columns=[label_column])

    print(f"Classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    X_train_scaled.to_csv(train_features_path, index=False)
    X_test_scaled.to_csv(test_features_path, index=False)
    y_train.to_csv(train_labels_path, index=False)
    y_test.to_csv(test_labels_path, index=False)

    print(f"Train: {len(X_train_scaled)}, Test: {len(X_test_scaled)}")


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
    packages_to_install=["scikit-learn==1.5.2", "pandas==2.2.3", "joblib==1.4.2"],
)
def train_model(
    train_features_path: dsl.InputPath("csv"),
    train_labels_path: dsl.InputPath("csv"),
    model_path: dsl.OutputPath("joblib"),
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
):
    """Train a RandomForest classifier on the training data."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"Model trained with {n_estimators} estimators, max_depth={max_depth}")


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
    packages_to_install=["scikit-learn==1.5.2", "pandas==2.2.3", "joblib==1.4.2"],
)
def evaluate_model(
    model_path: dsl.InputPath("joblib"),
    test_features_path: dsl.InputPath("csv"),
    test_labels_path: dsl.InputPath("csv"),
    metrics_path: dsl.OutputPath("json"),
) -> float:
    """Evaluate the trained model and return accuracy."""
    import json

    import joblib
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )

    model = joblib.load(model_path)
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_labels_path).squeeze()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1),
        "classification_report": report,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}, F1 (weighted): {f1:.4f}")
    return accuracy


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
    packages_to_install=["joblib==1.4.2"],
)
def export_model(
    model_path: dsl.InputPath("joblib"),
    exported_model_path: dsl.OutputPath("joblib"),
    accuracy: float,
):
    """Copy the model to the export path for downstream serving."""
    import shutil

    shutil.copy2(model_path, exported_model_path)
    print(f"Model exported (accuracy={accuracy:.4f}), ready for serving.")


@dsl.pipeline(
    name="Iris Classifier Training Pipeline",
    description=(
        "Fetch an API key from Vault, fetch data from a URI connection, "
        "train, evaluate, and conditionally export an Iris classifier on OpenShift AI."
    ),
)
def iris_training_pipeline(
    vault_uri: str = "https://vault-vault.apps.ocp.lgvzs.sandbox180.opentlc.com/v1/secret/data/apikey",
    vault_token: str = "",
    data_uri: str = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    label_column: str = "species",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 5,
    accuracy_threshold: float = 0.90,
):
    vault_task = fetch_secret_from_vault(
        vault_uri=vault_uri,
        vault_token=vault_token,
    )
    apply_kueue_config(vault_task)

    fetch_task = fetch_data_from_uri(data_uri=data_uri)
    apply_kueue_config(fetch_task)

    preprocess_task = preprocess_data(
        raw_data_path=fetch_task.outputs["raw_data_path"],
        label_column=label_column,
        test_size=test_size,
    )
    preprocess_task.after(vault_task)
    apply_kueue_config(preprocess_task)

    train_task = train_model(
        train_features_path=preprocess_task.outputs["train_features_path"],
        train_labels_path=preprocess_task.outputs["train_labels_path"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    apply_kueue_config(train_task)

    eval_task = evaluate_model(
        model_path=train_task.outputs["model_path"],
        test_features_path=preprocess_task.outputs["test_features_path"],
        test_labels_path=preprocess_task.outputs["test_labels_path"],
    )
    apply_kueue_config(eval_task)

    with dsl.If(eval_task.outputs["Output"] >= accuracy_threshold):
        export_task = export_model(
            model_path=train_task.outputs["model_path"],
            accuracy=eval_task.outputs["Output"],
        )
        apply_kueue_config(export_task)


if __name__ == "__main__":
    import pathlib

    output = pathlib.Path(__file__).with_name("iris_training_pipeline.yaml")
    compiler.Compiler().compile(
        pipeline_func=iris_training_pipeline,
        package_path=str(output),
    )
    print(f"Pipeline compiled to {output}")
