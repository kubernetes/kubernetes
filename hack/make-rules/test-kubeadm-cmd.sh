#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

KUBEADM_PATH="${KUBEADM_PATH:=$(realpath "${KUBE_ROOT}")/cluster/kubeadm.sh}"

# If testing a different version of kubeadm than the current build, you can
# comment this out to save yourself from needlessly building here.
make -C "${KUBE_ROOT}" WHAT=cmd/kubeadm

make -C "${KUBE_ROOT}" test \
  WHAT=cmd/kubeadm/app/cmd \
  KUBE_GOFLAGS="-tags cmd" \
  KUBE_TEST_ARGS="--kubeadm-path '${KUBEADM_PATH}'"
