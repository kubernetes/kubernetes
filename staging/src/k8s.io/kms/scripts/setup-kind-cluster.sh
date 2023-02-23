#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# input and default values
KUBERNETES_VERSION="${1:-v1.26.0}"
REGISTRY_NAME="${2:-kind-registry}"
REGISTRY_PORT="${3:-5000}"

# create a cluster with the local registry enabled in containerd
# add encryption config and the kms static pod manifest with custom image
cat <<EOF | kind create cluster --retain --image kindest/node:"${KUBERNETES_VERSION}" --name "kind" --wait 2m --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
- |-
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors."localhost:${REGISTRY_PORT}"]
    endpoint = ["http://${REGISTRY_NAME}:${REGISTRY_PORT}"]
nodes:
- role: control-plane
  extraMounts:
  - containerPath: /etc/kubernetes/encryption-config.yaml
    hostPath: scripts/encryption-config.yaml
    readOnly: true
    propagation: None
  - containerPath: /etc/kubernetes/manifests/kubernetes-kms.yaml
    hostPath: plugins/base64/kms.yaml
    readOnly: true
    propagation: None
  kubeadmConfigPatches:
    - |
      kind: ClusterConfiguration
      apiServer:
        extraArgs:
          encryption-provider-config: "/etc/kubernetes/encryption-config.yaml"
          feature-gates: "KMSv2=true"
        extraVolumes:
        - name: encryption-config
          hostPath: "/etc/kubernetes/encryption-config.yaml"
          mountPath: "/etc/kubernetes/encryption-config.yaml"
          readOnly: true
          pathType: File
        - name: sock-path
          hostPath: "/tmp"
          mountPath: "/tmp"
EOF
