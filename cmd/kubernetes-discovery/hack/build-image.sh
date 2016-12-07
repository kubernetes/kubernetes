#!/bin/bash


KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../..
source "${KUBE_ROOT}/hack/lib/util.sh"

# Register function to be called on EXIT to remove generated binary.
function cleanup {
  rm "${KUBE_ROOT}/cmd/kubernetes-discovery/artifacts/simple-image/kubernetes-discovery"
}
trap cleanup EXIT

cp -v $(kube::util::find-binary kubernetes-discovery) "${KUBE_ROOT}/cmd/kubernetes-discovery/artifacts/simple-image/kubernetes-discovery"
docker build -t kubernetes-discovery:latest ${KUBE_ROOT}/cmd/kubernetes-discovery/artifacts/simple-image