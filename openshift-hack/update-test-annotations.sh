#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

# Update e2e test annotations that indicate openshift compatibility
GO111MODULE=on go generate -mod vendor ./openshift-hack/e2e
