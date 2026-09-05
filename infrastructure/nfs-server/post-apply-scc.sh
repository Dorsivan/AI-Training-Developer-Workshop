#!/bin/bash
# Run after applying nfs-server.yaml to grant required SCCs
oc adm policy add-scc-to-user anyuid -z nfs-sa -n nfs-server
oc adm policy add-scc-to-user privileged -z nfs-sa -n nfs-server
oc rollout restart deployment/nfs-ganesha -n nfs-server
oc rollout status deployment/nfs-ganesha -n nfs-server --timeout=120s
