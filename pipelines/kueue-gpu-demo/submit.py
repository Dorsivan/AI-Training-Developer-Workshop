"""Submit the Kueue demo pipeline.

Usage:
    python submit.py team-a                  # direct pod integration (with retry patch)
    python submit.py team-b                  # direct pod integration (with retry patch)
    python submit.py team-a --trainjob       # TrainJob pattern (no patch needed)
    python submit.py team-b --trainjob       # TrainJob pattern (no patch needed)
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
        "trainjob_yaml": "kueue_trainjob_pipeline.yaml",
    },
    "team-b": {
        "endpoint": "https://ds-pipeline-dspa-second-ai-project.apps.ocp.local",
        "namespace": "second-ai-project",
        "pipeline_yaml": "kueue_demo_pipeline_team_b.yaml",
        "trainjob_yaml": "kueue_trainjob_pipeline_team_b.yaml",
    },
}


def _patch_argo_retry_policy(namespace: str, run_id: str, policy: str = "Always",
                              timeout: int = 30, interval: int = 3):
    """Patch Argo Workflow to add retryPolicy after KFP creates it.

    KFP's set_retry() generates Argo retryStrategy with limit/backoff
    but no retryPolicy. Argo defaults to OnFailure, which doesn't
    catch Kueue preemptions (pod deletion = Error, not Failure).
    """
    import json
    import time

    label = f"pipeline/runid={run_id}"
    deadline = time.time() + timeout

    while time.time() < deadline:
        result = subprocess.run(
            ["oc", "get", "workflows", "-n", namespace, "-l", label,
             "-o", "jsonpath={.items[0].metadata.name}"],
            capture_output=True, text=True,
        )
        wf_name = result.stdout.strip()
        if wf_name:
            break
        time.sleep(interval)
    else:
        print(f"  [warn] Timed out waiting for Argo Workflow (label: {label})")
        return False

    result = subprocess.run(
        ["oc", "get", "workflow", wf_name, "-n", namespace, "-o", "json"],
        capture_output=True, text=True,
    )
    wf = json.loads(result.stdout)

    patches = []
    for i, tmpl in enumerate(wf.get("spec", {}).get("templates", [])):
        if tmpl.get("retryStrategy") and "retryPolicy" not in tmpl["retryStrategy"]:
            patches.append({
                "op": "add",
                "path": f"/spec/templates/{i}/retryStrategy/retryPolicy",
                "value": policy,
            })

    if not patches:
        print(f"  [info] No retryStrategy templates found in {wf_name}")
        return False

    subprocess.run(
        ["oc", "patch", "workflow", wf_name, "-n", namespace,
         "--type=json", "-p", json.dumps(patches)],
        check=True,
    )
    print(f"  [ok] Patched {wf_name}: retryPolicy={policy} ({len(patches)} template(s))")
    return True


def main():
    args = sys.argv[1:]
    use_trainjob = "--trainjob" in args
    args = [a for a in args if a != "--trainjob"]
    team = args[0] if args else "team-a"

    if team not in TEAMS:
        print(f"Unknown team: {team}. Choose from: {', '.join(TEAMS)}")
        sys.exit(1)

    cfg = TEAMS[team]
    yaml_key = "trainjob_yaml" if use_trainjob else "pipeline_yaml"
    yaml_path = pathlib.Path(__file__).with_name(cfg[yaml_key])
    mode = "trainjob" if use_trainjob else "direct-pod"

    token = subprocess.check_output(["oc", "whoami", "-t"]).decode().strip()
    client = kfp.Client(
        host=cfg["endpoint"], existing_token=token, namespace=cfg["namespace"],
    )
    run = client.create_run_from_pipeline_package(
        str(yaml_path),
        experiment_name="kueue-demo",
        run_name=f"kueue-demo-{team}-{mode}",
        namespace=cfg["namespace"],
    )
    print(f"[{team}] Run submitted ({mode}): {run.run_id}")

    if not use_trainjob:
        _patch_argo_retry_policy(cfg["namespace"], run.run_id)
    else:
        print("  [info] TrainJob pattern — no Argo retry patch needed")


if __name__ == "__main__":
    main()
