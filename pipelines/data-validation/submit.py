"""Upload a new pipeline version and start a run.

Usage:
    python submit.py
"""

import pathlib
import subprocess

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
PIPELINE_YAML = "data_validation_pipeline.yaml"


def main():
    yaml_path = pathlib.Path(__file__).with_name(PIPELINE_YAML)

    token = subprocess.check_output(["oc", "whoami", "-t"]).decode().strip()
    client = kfp.Client(host=ENDPOINT, existing_token=token, namespace=NAMESPACE)
    run = client.create_run_from_pipeline_package(
        str(yaml_path),
        experiment_name="data-validation",
        run_name="data-validation",
        namespace=NAMESPACE,
    )
    print(f"Run submitted: {run.run_id}")


if __name__ == "__main__":
    main()
