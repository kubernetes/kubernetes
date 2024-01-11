#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script verifies the build without in-tree cloud providers.
# Usage: `hack/verify-typecheck-providerless.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

cd "${KUBE_ROOT}"
# verify the providerless build
# https://github.com/kubernetes/enhancements/blob/master/keps/sig-cloud-provider/1179-building-without-in-tree-providers/README.md
hack/verify-typecheck.sh --skip-test --tags=providerless

# verify using go list
if _out="$(go list -mod=readonly -tags "providerless" -e -json  k8s.io/kubernetes/cmd/kubelet/... \
  | grep -e Azure/azure-sdk-for-go -e github.com/aws/aws-sdk-go -e google.golang.org/api)"; then
    echo "${_out}" >&2
    echo "Verify typecheck for providerless tag failed. Found restricted packages." >&2
    exit 1
fi
if _out="$(go list -mod=readonly -tags "providerless" -e -json  k8s.io/kubernetes/cmd/kube-apiserver/... \
  | grep -e Azure/azure-sdk-for-go -e github.com/aws/aws-sdk-go -e google.golang.org/api \
         -e Azure/go-autorest -e oauth2/google)"; then
    echo "${_out}" >&2
    echo "Verify typecheck for providerless tag failed. Found restricted packages." >&2
    exit 1
fi
