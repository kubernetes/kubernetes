#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Update kubensenter and error if a change is detected
"${KUBE_ROOT}"/hack/update-kubensenter.sh
git diff --quiet "${KUBE_ROOT}/openshift-hack/images/hyperkube/kubensenter"
