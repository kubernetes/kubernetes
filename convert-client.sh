#!/bin/bash
# convert pkg/client/ to versioned API
set -o errexit
set -o nounset
set -o pipefail
KUBE_ROOT=$(dirname "${BASH_SOURCE}")
echo "converting pkg/client/record to v1"
# need a v1 version of ref.go
cp "${KUBE_ROOT}"/pkg/api/ref.go "${KUBE_ROOT}"/pkg/api/v1/ref.go
gofmt -w -r 'api.a -> v1.a' "${KUBE_ROOT}"/pkg/api/v1/ref.go
gofmt -w -r 'Scheme -> api.Scheme' "${KUBE_ROOT}"/pkg/api/v1/ref.go
# rewriting package name to v1
sed -i 's/package api/package v1/g' "${KUBE_ROOT}"/pkg/api/v1/ref.go
# ref.go refers api.Scheme, so manually import /pkg/api
sed -i "s,import (,import (\n\"k8s.io/kubernetes/pkg/api\",g" "${KUBE_ROOT}"/pkg/api/v1/ref.go
gofmt -w "${KUBE_ROOT}"/pkg/api/v1/ref.go 
# rewrite pkg/client/record to v1
gofmt -w -r 'api.a -> v1.a' "${KUBE_ROOT}"/pkg/client/record
# need to call sed to rewrite the strings in test cases...
find "${KUBE_ROOT}"/pkg/client/record -type f -name "*.go" -print0 | xargs -0 sed -i "s/api.ObjectReference/v1.ObjectReference/g"
# rewrite the imports
find "${KUBE_ROOT}"/pkg/client/record -type f -name "*.go" -print0 | xargs -0 sed -i 's,pkg/api",pkg/api/v1",g'

echo "converting pkg/client/cache to use versioned objects"
readonly CACHE_DIR="${KUBE_ROOT}/pkg/client/cache"
gofmt -w -r 'api.a -> v1.a' "${CACHE_DIR}"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|v1\.ParameterCodec|api\.ParameterCodec|g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|v1\.Scheme|api\.Scheme|g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/api\",\"k8s.io/kubernetes/pkg/api/v1\"\n  \"k8s.io/kubernetes/pkg/api\",g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/apis/apps\",apps \"k8s.io/kubernetes/pkg/apis/apps/v1alpha1\",g"
# maybe we need v2alpha1
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/apis/batch\",batch \"k8s.io/kubernetes/pkg/apis/batch/v1\",g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/apis/certificates\",certificates \"k8s.io/kubernetes/pkg/apis/certificates/v1alpha1\",g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/apis/extensions\",extensions \"k8s.io/kubernetes/pkg/apis/extensions/v1beta1\"\n    extensionsinternal \"k8s.io/kubernetes/pkg/apis/extensions\",g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/apis/policy\",policy \"k8s.io/kubernetes/pkg/apis/policy/v1alpha1\",g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/apis/rbac\",rbac \"k8s.io/kubernetes/pkg/apis/rbac/v1alpha1\"\n   rbacinternal \"k8s.io/kubernetes/pkg/apis/rbac\",g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/apis/storage\",storage \"k8s.io/kubernetes/pkg/apis/storage/v1beta1\"\n   storageinternal \"k8s.io/kubernetes/pkg/apis/storage\",g"

# corner cases
# TODO: These are wrong. we should move Resource() to a util package
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|v1\.Resource(|api\.Resource(|g"
#find "${CACHE_DIR}" -type f \( -name "*listers_core.go" -o -name "*listwatch.go" \) -print0 | xargs -0 sed -i "s,\"k8s.io/kubernetes/pkg/v1\",\"k8s.io/kubernetes/pkg/api/v1\"\n    \"k8s.io/kubernetes/pkg/api\",g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|rbac\.Resource(|rbacinternal\.Resource(|g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|extensions\.Resource(|extensionsinternal\.Resource(|g"
find "${CACHE_DIR}" -type f -name "*.go" -print0 | xargs -0 sed -i "s|storage\.Resource(|storageinternal\.Resource(|g"
goimports -w "${CACHE_DIR}"
