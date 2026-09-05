"""
Data Validation & Processing Pipeline for Red Hat OpenShift AI.

This pipeline demonstrates data validation with conditional branching:
  1. Ingest synthetic data and write to a shared PVC
  2. Validate data quality (check nulls, duplicates, row count)
  3. If quality is good: transform the data and generate a report
  4. If quality is bad: quarantine the data and send an alert

Prerequisites:
  - A PVC named 'pipeline-shared-data' must exist in the namespace.
    See pipeline-pvc.yaml for the manifest.

Compile with:
    python data_validation_pipeline.py

Then upload the generated YAML to the OpenShift AI dashboard.
"""

from kfp import compiler, dsl
from kfp import kubernetes

# Kueue scheduling configuration (compile-time values — recompile to change).
KUEUE_LOCAL_QUEUE = "pipeline-local-queue"
KUEUE_PRIORITY = "pipeline-default-priority"


def apply_kueue_config(task, queue_name=KUEUE_LOCAL_QUEUE, priority_class=KUEUE_PRIORITY):
    """Configure a pipeline task for Kueue scheduling and preemption."""
    kubernetes.add_pod_label(task, "kueue.x-k8s.io/queue-name", queue_name)
    kubernetes.add_pod_label(task, "kueue.x-k8s.io/priority-class", priority_class)


def apply_resource_config(task, cpu, memory, gpu):
    """Set CPU/memory requests+limits and optional GPU limit on a task."""
    task.set_cpu_request(cpu)
    task.set_cpu_limit(cpu)
    task.set_memory_request(memory)
    task.set_memory_limit(memory)
    if gpu and gpu != "0":
        task.set_gpu_limit(gpu)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
)
def ingest_data(num_rows: int, null_fraction: float) -> str:
    """Generate a synthetic dataset and write it to the shared PVC."""
    import csv
    import random

    random.seed(42)

    headers = ["id", "name", "value", "category"]
    categories = ["A", "B", "C", "D"]
    names = [
        "Alice", "Bob", "Charlie", "Diana",
        "Eve", "Frank", "Grace", "Hank",
    ]

    output_path = "/mnt/shared/raw_data.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(1, num_rows + 1):
            row = [
                i,
                random.choice(names),
                round(random.uniform(1.0, 100.0), 2),
                random.choice(categories),
            ]
            # Randomly inject null values to make validation interesting.
            for col_idx in range(len(row)):
                if random.random() < null_fraction:
                    row[col_idx] = ""
            writer.writerow(row)

    print(f"Ingested {num_rows} rows to {output_path}")
    return output_path


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
)
def validate_data(data_path: str) -> float:
    """Read raw data from the PVC, compute a quality score (0.0 - 1.0).

    Checks performed:
      - Null / empty cell count
      - Duplicate row count
      - Minimum row-count threshold (>= 10)

    Writes a validation report to /mnt/shared/validation_report.json.
    Returns the quality_score as a float so it can drive conditional branching.
    """
    import csv
    import json

    input_path = "/mnt/shared/raw_data.csv"

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    if total_rows == 0:
        print("No data found!")
        quality_score = 0.0
        report = {
            "total_rows": 0,
            "null_count": 0,
            "duplicate_count": 0,
            "quality_score": quality_score,
        }
        with open("/mnt/shared/validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return quality_score

    # Count null / empty cells.
    null_count = 0
    total_cells = 0
    for row in rows:
        for value in row.values():
            total_cells += 1
            if value is None or value.strip() == "":
                null_count += 1

    # Count duplicate rows (by all column values).
    seen: set = set()
    duplicate_count = 0
    for row in rows:
        row_tuple = tuple(row.values())
        if row_tuple in seen:
            duplicate_count += 1
        seen.add(row_tuple)

    # Quality score: penalise nulls and duplicates, reward sufficient rows.
    null_penalty = null_count / total_cells if total_cells > 0 else 0
    duplicate_penalty = duplicate_count / total_rows if total_rows > 0 else 0
    row_count_factor = 1.0 if total_rows >= 10 else total_rows / 10.0

    quality_score = max(
        0.0,
        min(1.0, (1.0 - null_penalty - duplicate_penalty) * row_count_factor),
    )

    report = {
        "total_rows": total_rows,
        "total_cells": total_cells,
        "null_count": null_count,
        "duplicate_count": duplicate_count,
        "null_rate": round(null_penalty, 4),
        "duplicate_rate": round(duplicate_penalty, 4),
        "quality_score": round(quality_score, 4),
    }

    with open("/mnt/shared/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation complete: quality_score={quality_score:.4f}")
    print(
        f"  Total rows: {total_rows}, Nulls: {null_count}, "
        f"Duplicates: {duplicate_count}"
    )
    return quality_score


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
)
def transform_data() -> int:
    """Clean the raw data: drop rows with nulls, remove duplicates.

    Reads  /mnt/shared/raw_data.csv
    Writes /mnt/shared/clean_data.csv
    Returns the number of clean rows.
    """
    import csv

    input_path = "/mnt/shared/raw_data.csv"
    output_path = "/mnt/shared/clean_data.csv"

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    # Drop rows with any null / empty values.
    clean_rows = []
    for row in rows:
        if all(v is not None and v.strip() != "" for v in row.values()):
            clean_rows.append(row)

    # Remove duplicate rows.
    seen: set = set()
    unique_rows = []
    for row in clean_rows:
        row_tuple = tuple(row.values())
        if row_tuple not in seen:
            unique_rows.append(row)
            seen.add(row_tuple)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"Transformed data: {len(rows)} -> {len(unique_rows)} rows")
    print(f"Written to {output_path}")
    return len(unique_rows)


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
)
def generate_report(row_count: int) -> str:
    """Generate a summary report from the clean data.

    Reads  /mnt/shared/clean_data.csv
    Writes /mnt/shared/report.json
    Returns the summary as a JSON string.
    """
    import csv
    import json

    input_path = "/mnt/shared/clean_data.csv"
    output_path = "/mnt/shared/report.json"

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Compute summary statistics on the 'value' column.
    values = []
    categories: dict = {}
    for row in rows:
        try:
            val = float(row["value"])
            values.append(val)
        except (ValueError, KeyError):
            pass
        cat = row.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    summary = {
        "total_clean_rows": len(rows),
        "rows_after_transform": row_count,
        "value_stats": {
            "min": round(min(values), 2) if values else None,
            "max": round(max(values), 2) if values else None,
            "mean": round(sum(values) / len(values), 2) if values else None,
        },
        "category_distribution": categories,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    summary_str = json.dumps(summary)
    print(f"Report generated: {summary_str}")
    return summary_str


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
)
def quarantine_data() -> str:
    """Copy raw data to a quarantine directory on the PVC.

    Reads  /mnt/shared/raw_data.csv
    Writes /mnt/shared/quarantine/raw_data.csv
    Returns the quarantine file path.
    """
    import os
    import shutil

    source_path = "/mnt/shared/raw_data.csv"
    quarantine_dir = "/mnt/shared/quarantine"
    os.makedirs(quarantine_dir, exist_ok=True)

    quarantine_path = os.path.join(quarantine_dir, "raw_data.csv")
    shutil.copy2(source_path, quarantine_path)

    print(f"Data quarantined to {quarantine_path}")
    return quarantine_path


@dsl.component(
    base_image="registry.redhat.io/ubi9/python-311:latest",
)
def send_alert(quarantine_path: str) -> str:
    """Log an alert about quarantined data and return the alert message."""
    alert_message = (
        f"ALERT: Data quality check FAILED. "
        f"Raw data has been quarantined to: {quarantine_path}. "
        f"Please review the data and re-submit after fixing quality issues."
    )
    print("=" * 60)
    print("ALERT NOTIFICATION")
    print("=" * 60)
    print(alert_message)
    print("=" * 60)
    return alert_message


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dsl.pipeline(
    name="Data Validation and Processing Pipeline",
    description=(
        "Validates ingested data quality and conditionally transforms good data "
        "or quarantines bad data with an alert notification. "
        "Dynamically provisions a PVC via CreatePVC and cleans up with DeletePVC."
    ),
)
def data_validation_pipeline(
    quality_threshold: float = 0.8,
    num_rows: int = 100,
    null_fraction: float = 0.05,
    pvc_size: str = "1Gi",
    storage_class: str = "",
    cpu: str = "500m",
    memory: str = "512Mi",
    gpu: str = "0",
):
    # Step 0: Dynamically create a PVC for sharing data between steps.
    create_pvc_task = kubernetes.CreatePVC(
        pvc_name_suffix="-shared-data",
        access_modes=["ReadWriteOnce"],
        size=pvc_size,
        storage_class_name=storage_class if storage_class else None,
    )

    # Step 1: Ingest synthetic data to the shared PVC.
    ingest_task = ingest_data(
        num_rows=num_rows,
        null_fraction=null_fraction,
    )
    ingest_task.after(create_pvc_task)
    kubernetes.mount_pvc(
        ingest_task,
        pvc_name=create_pvc_task.outputs["name"],
        mount_path="/mnt/shared",
    )
    apply_kueue_config(ingest_task)
    apply_resource_config(ingest_task, cpu, memory, gpu)

    # Step 2: Validate data quality (reads from PVC, returns quality_score).
    validate_task = validate_data(data_path=ingest_task.output)
    kubernetes.mount_pvc(
        validate_task,
        pvc_name=create_pvc_task.outputs["name"],
        mount_path="/mnt/shared",
    )
    apply_kueue_config(validate_task)
    apply_resource_config(validate_task, cpu, memory, gpu)

    # Step 3: Conditional branching based on quality score.
    with dsl.If(validate_task.output >= quality_threshold):
        # --- Good-data path ---
        transform_task = transform_data()
        kubernetes.mount_pvc(
            transform_task,
            pvc_name=create_pvc_task.outputs["name"],
            mount_path="/mnt/shared",
        )
        apply_kueue_config(transform_task)
        apply_resource_config(transform_task, cpu, memory, gpu)

        report_task = generate_report(row_count=transform_task.output)
        kubernetes.mount_pvc(
            report_task,
            pvc_name=create_pvc_task.outputs["name"],
            mount_path="/mnt/shared",
        )
        apply_kueue_config(report_task)
        apply_resource_config(report_task, cpu, memory, gpu)

    with dsl.Else():
        # --- Bad-data path ---
        quarantine_task = quarantine_data()
        kubernetes.mount_pvc(
            quarantine_task,
            pvc_name=create_pvc_task.outputs["name"],
            mount_path="/mnt/shared",
        )
        apply_kueue_config(quarantine_task)
        apply_resource_config(quarantine_task, cpu, memory, gpu)

        alert_task = send_alert(quarantine_path=quarantine_task.output)
        apply_kueue_config(alert_task)
        apply_resource_config(alert_task, cpu, memory, gpu)

    # Step 4: Clean up the PVC after all branches complete.
    delete_pvc_task = kubernetes.DeletePVC(
        pvc_name=create_pvc_task.outputs["name"],
    )
    delete_pvc_task.after(report_task, alert_task)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib

    output = pathlib.Path(__file__).with_name("data_validation_pipeline.yaml")
    compiler.Compiler().compile(
        pipeline_func=data_validation_pipeline,
        package_path=str(output),
    )
    print(f"Pipeline compiled to {output}")
