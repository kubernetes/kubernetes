#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Make sure that all packages that define k8s tests are properly imported
EXCLUDE_PACKAGES="\
k8s.io/kubernetes/test/e2e/framework,\
k8s.io/kubernetes/test/e2e/framework/debug/init,\
k8s.io/kubernetes/test/e2e/framework/metrics/init,\
k8s.io/kubernetes/test/e2e/framework/node/init,\
k8s.io/kubernetes/test/e2e/framework/testfiles,\
k8s.io/kubernetes/test/e2e/storage/external,\
k8s.io/kubernetes/test/e2e/testing-manifests,\
k8s.io/kubernetes/test/e2e/windows"

GO111MODULE=on go run ./openshift-hack/cmd/go-imports-diff \
    -exclude "$EXCLUDE_PACKAGES" \
    test/e2e/e2e_test.go \
    openshift-hack/e2e/include.go

# Verify e2e test annotations that indicate openshift compatibility
"${KUBE_ROOT}"/hack/update-test-annotations.sh
git diff --quiet "${KUBE_ROOT}/openshift-hack/e2e/annotate/generated/"
