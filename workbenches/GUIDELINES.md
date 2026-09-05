## Guidelines To Build a Good Workbench Image

Usually, we would build workbench images for one of two reasons:

1. We wish to pre-install packages / configurations that need to exist in every workbench, or that cannot be installed without root permissions.
2. We have a specific unique image we want to transform into a workbench.

### Creating a custom image from a default OpenShift AI image

Start from one of the standard OpenShift AI notebook images and add your packages on top:

```dockerfile
FROM image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/jupyter-datascience-notebook:2024.1

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
are: expose port 8888, serve `/api` for readiness/liveness probes, and run as
USER 1001.

```dockerfile
FROM registry.redhat.io/ubi9/python-311:latest

USER 0

# Install system packages.
RUN dnf install -y gcc python3-devel && \
    dnf clean all

# Install JupyterLab and your dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir jupyterlab -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create the workspace directory and make it group-writable.
RUN mkdir -p /opt/app-root/src && \
    chgrp -R 0 /opt/app-root && chmod -R g=u /opt/app-root

USER 1001
WORKDIR /opt/app-root/src

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--NotebookApp.token=''", \
     "--NotebookApp.base_url=/", \
     "--ServerApp.allow_origin='*'"]
```

Use the Red Hat UBI

Designing your image to run with USER 1001

In OpenShift, your container will run with a random UID and a GID of 0. Make sure that your image is compatible with these user and group requirements, especially if you need write access to directories. Best practice is to design your image to run with USER 1001.

Avoid placing artifacts in $HOME

The persistent volume attached to the workbench will be mounted on /opt/app-root/src. This location is also the location of $HOME. Therefore, do not put any files or other resources directly in $HOME because they won’t be visible after the workbench is deployed (and the persistent volume is mounted).

Specifying the API endpoint

OpenShift readiness and liveness probes will query the /api endpoint. For a Jupyter IDE, this is the default endpoint. For other IDEs, you must implement the /api endpoint.

### Best Practices

- Do not assume a fixed UID at runtime. OpenShift commonly runs containers with an arbitrary UID under its SCC model. Writable directories should be owned by group 0 and have group permissions equivalent to user permissions, e.g. chgrp -R 0 ... && chmod -R g=u ....
- Do not require root at runtime. Install RPMs/system libraries while building the image, then switch back to the notebook user. Red Hat's own custom-workbench examples install system packages as root and then use USER 1001 for Python packages/runtime.
- Avoid writing into arbitrary system paths at runtime. Anything the notebook or Python stack needs to modify should be under writable locations such as the user's home, workspace, /tmp, or another explicitly prepared directory.
- Do not listen on privileged ports <1024. Arbitrary non-root UIDs cannot bind them.
Be careful with software that requires /etc/passwd lookup. An OpenShift-assigned UID may not exist in /etc/passwd; some applications break on whoami, home-directory resolution, etc. Red Hat documents the dynamic passwd-entry pattern for software that requires it.
- Bake stable dependencies into the image. Installing packages manually from Jupyter is fine experimentally, but those changes are not a reliable/reproducible environment after restart. Custom images are the intended solution for shared/stable dependencies.