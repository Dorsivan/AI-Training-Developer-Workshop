"""Submit the Kueue CPU demo pipeline.

Usage:
    python submit.py                # submit to first-ai-project (team-a)
    python submit.py team-b         # submit to second-ai-project (team-b)
"""

import pathlib
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

TEAMS = {
    "team-a": {
        "endpoint": "https://ds-pipeline-dspa-first-ai-project.apps.ocp.local",
        "namespace": "first-ai-project",
        "pipeline_yaml": "kueue_demo_pipeline.yaml",
    },
    "team-b": {
        "endpoint": "https://ds-pipeline-dspa-second-ai-project.apps.ocp.local",
        "namespace": "second-ai-project",
        "pipeline_yaml": "kueue_demo_pipeline_team_b.yaml",
    },
}


def main():
    team = sys.argv[1] if len(sys.argv) > 1 else "team-a"
    if team not in TEAMS:
        print(f"Unknown team: {team}. Choose from: {', '.join(TEAMS)}")
        sys.exit(1)

    cfg = TEAMS[team]
    yaml_path = pathlib.Path(__file__).with_name(cfg["pipeline_yaml"])

    token = subprocess.check_output(["oc", "whoami", "-t"]).decode().strip()
    client = kfp.Client(
        host=cfg["endpoint"], existing_token=token, namespace=cfg["namespace"],
    )
    run = client.create_run_from_pipeline_package(
        str(yaml_path),
        experiment_name="kueue-cpu-demo",
        run_name=f"kueue-cpu-demo-{team}",
        namespace=cfg["namespace"],
    )
    print(f"[{team}] Run submitted: {run.run_id}")


if __name__ == "__main__":
    main()
