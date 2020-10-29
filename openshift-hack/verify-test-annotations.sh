#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Verify e2e test annotations that indicate openshift compatibility
"${KUBE_ROOT}"/hack/update-test-annotations.sh
git diff --quiet "${KUBE_ROOT}/openshift-hack/e2e/annotate/generated/"
