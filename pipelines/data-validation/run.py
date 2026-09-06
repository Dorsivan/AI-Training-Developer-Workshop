"""Create a new run from an existing pipeline version.

Usage:
    python run.py                          # runs the latest version
    python run.py <version-id>             # runs a specific version
    python run.py --list                   # lists available versions
"""

import argparse
import subprocess
import sys

import urllib3
urllib3.disable_warnings()

import kfp_server_api

_orig_init = kfp_server_api.Configuration.__init__
def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.verify_ssl = False
kfp_server_api.Configuration.__init__ = _patched_init

import kfp

ENDPOINT = "https://ds-pipeline-dspa-first-ai-project.apps.ocp.local"
NAMESPACE = "first-ai-project"
EXPERIMENT = "data-validation"
PIPELINE_NAME = "data_validation_pipeline"


def get_client():
    token = subprocess.check_output(["oc", "whoami", "-t"]).decode().strip()
    return kfp.Client(host=ENDPOINT, existing_token=token, namespace=NAMESPACE)


def find_pipeline(client):
    pipelines = client.list_pipelines()
    for p in pipelines.pipelines or []:
        if p.display_name == PIPELINE_NAME:
            return p.pipeline_id
    print(f"Pipeline '{PIPELINE_NAME}' not found. Upload one first with submit.py")
    sys.exit(1)


def list_versions(client, pipeline_id):
    versions = client.list_pipeline_versions(pipeline_id)
    for v in versions.pipeline_versions or []:
        print(f"  {v.pipeline_version_id}  {v.display_name}  created={v.created_at}")


def main():
    parser = argparse.ArgumentParser(description="Run an existing pipeline version")
    parser.add_argument("version_id", nargs="?", default=None, help="Version ID (default: latest)")
    parser.add_argument("--list", action="store_true", help="List available versions")
    parser.add_argument("--name", default="data-validation-run", help="Run display name")
    args = parser.parse_args()

    client = get_client()
    pipeline_id = find_pipeline(client)

    if args.list:
        print(f"Versions for pipeline {pipeline_id}:")
        list_versions(client, pipeline_id)
        return

    version_id = args.version_id
    if not version_id:
        versions = client.list_pipeline_versions(pipeline_id)
        if not versions.pipeline_versions:
            print("No versions found. Upload one first with submit.py")
            sys.exit(1)
        version_id = versions.pipeline_versions[0].pipeline_version_id
        print(f"Using latest version: {version_id}")

    experiment = client.create_experiment(name=EXPERIMENT, namespace=NAMESPACE)
    run = client.run_pipeline(
        experiment_id=experiment.experiment_id,
        job_name=args.name,
        version_id=version_id,
    )
    print(f"Run submitted: {run.run_id}")


if __name__ == "__main__":
    main()
