## Guidelines To Build a Good Workbench Image

Usually, we would build workbench images for one of two reasons:

1. We wish to pre-install packages / configurations that need to exist in every workbench, or that cannot be installed without root permissions.
2. We have a specific unique image we want to transform into a workbench.

### Creating a custom image from a default OpenShift AI image

Start from one of the standard OpenShift AI notebook images and add your packages on top.
The base image already includes the correct entrypoint (`start-notebook.sh`), port (8888),
and directory structure — so you only need to add your customizations:

```dockerfile
FROM quay.io/modh/odh-generic-data-science-notebook@sha256:<digest>

USER 0

# Install system-level dependencies that require root.
RUN dnf install -y git-lfs && \
    dnf clean all

# Install Python packages.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Ensure any new directories are group-writable for OpenShift's arbitrary UID.
RUN chgrp -R 0 /opt/app-root && chmod -R g=u /opt/app-root

USER 1001
```

### Creating a custom image from your own image

Start from Red Hat UBI and add the Jupyter stack yourself. The key requirements
for RHOAI compatibility are listed in the section below.

```dockerfile
FROM registry.access.redhat.com/ubi9/python-311:latest

USER 0

RUN dnf install -y gcc python3-devel procps-ng && \
    dnf clean all

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN mkdir -p /opt/app-root/etc/jupyter /opt/app-root/bin
COPY jupyter_server_config.py /opt/app-root/etc/jupyter/
COPY start-notebook.sh /opt/app-root/bin/

RUN chmod +x /opt/app-root/bin/start-notebook.sh && \
    mkdir -p /opt/app-root/src && \
    chgrp -R 0 /opt/app-root && chmod -R g=u /opt/app-root

ENV HOME=/opt/app-root/src \
    JUPYTER_CONFIG_DIR=/opt/app-root/etc/jupyter \
    APP_ROOT=/opt/app-root \
    PATH=/opt/app-root/src/.local/bin:/opt/app-root/bin:$PATH

USER 1001
WORKDIR /opt/app-root/src

EXPOSE 8080

ENTRYPOINT ["start-notebook.sh"]
```

### RHOAI Workbench Image Requirements

The RHOAI Notebook controller expects custom images to follow specific
conventions. Failing to meet them results in broken probes, missing auth
sidecars, or no routing.

**Entrypoint — `start-notebook.sh`:**
The image must use `start-notebook.sh` as its entrypoint, not a hardcoded
`CMD ["jupyter", "lab", ...]`. The controller passes configuration via the
`NOTEBOOK_ARGS` env var (base_url, port, token settings). A hardcoded CMD
ignores `NOTEBOOK_ARGS` and causes base_url mismatches. See
`custom-jupyter/start-notebook.sh` for a minimal implementation.

The `start-notebook.sh` script must NOT set a default `--ServerApp.port`
because `NOTEBOOK_ARGS` already includes the port. Duplicate port arguments
cause Jupyter to fail with:
`Bad config encountered: port only accepts one value, got 2`.

**Port — 8888 (not 8080):**
The standard RHOAI images serve on port 8888. The `NOTEBOOK_ARGS` env var
set by the controller specifies `--ServerApp.port=8888`. The container should
expose port 8888 and the Notebook CR should set `containerPort: 8888`.

**`/api` endpoint:**
The readiness and liveness probes query `/notebook/<namespace>/<name>/api`.
JupyterLab serves this by default when the base_url is set correctly via
`NOTEBOOK_ARGS`.

### Deploying Workbenches via CLI (Notebook CR)

When creating a Notebook CR directly (not through the RHOAI dashboard), the
following metadata is required for the controller to properly reconcile the
workbench (inject the auth sidecar, create routing, etc.):

**Labels (required):**
```yaml
labels:
  app: <notebook-name>
  opendatahub.io/dashboard: "true"
  opendatahub.io/odh-managed: "true"
```

**Annotations (required):**
```yaml
annotations:
  notebooks.opendatahub.io/inject-auth: "true"
  opendatahub.io/image-display-name: "<display name>"
  opendatahub.io/username: "<user>"
```

**Important: `inject-auth`, not `inject-oauth`.**
RHOAI 3.x uses a `kube-rbac-proxy` sidecar for authentication, not the
legacy OAuth proxy. The annotation is `inject-auth: "true"`. Using the old
`inject-oauth: "true"` annotation will NOT inject the sidecar, leaving the
workbench without authentication and without routing through the RHOAI
gateway.

**The `NOTEBOOK_ARGS` env var:**
The controller sets `NOTEBOOK_ARGS` on the container with the correct
`base_url`, `port`, and auth settings. The image entrypoint must read and
apply these arguments. Example value set by the controller:
```
--ServerApp.port=8888
--ServerApp.token=''
--ServerApp.password=''
--ServerApp.base_url=/notebook/<namespace>/<name>
--ServerApp.quit_button=False
```

**Routing:**
The RHOAI controller creates HTTPRoutes through the RHOAI gateway
(`rh-ai.apps.<cluster>`). You do not need to create OpenShift Routes
manually. If HTTPRoutes are not created, check that the labels and
annotations above are correct.

### Best Practices

- Do not assume a fixed UID at runtime. OpenShift commonly runs containers with an arbitrary UID under its SCC model. Writable directories should be owned by group 0 and have group permissions equivalent to user permissions, e.g. `chgrp -R 0 ... && chmod -R g=u ...`.
- Do not require root at runtime. Install RPMs/system libraries while building the image, then switch back to the notebook user.
- Avoid writing into arbitrary system paths at runtime. Anything the notebook or Python stack needs to modify should be under writable locations such as the user's home, workspace, `/tmp`, or another explicitly prepared directory.
- Do not listen on privileged ports <1024. Arbitrary non-root UIDs cannot bind them.
- Be careful with software that requires `/etc/passwd` lookup. An OpenShift-assigned UID may not exist in `/etc/passwd`; some applications break on `whoami`, home-directory resolution, etc.
- Bake stable dependencies into the image. Installing packages manually from Jupyter is fine experimentally, but those changes are not a reliable/reproducible environment after restart. Custom images are the intended solution for shared/stable dependencies.
- Use `registry.access.redhat.com` (no auth) instead of `registry.redhat.io` (requires auth) for UBI base images when possible.
