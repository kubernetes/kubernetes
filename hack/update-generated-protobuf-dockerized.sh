#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

BINS=(
	vendor/k8s.io/code-generator/cmd/go-to-protobuf
	vendor/k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo
)
make -C "${KUBE_ROOT}" WHAT="${BINS[*]}"

if [[ -z "$(which protoc)" || "$(protoc --version)" != "libprotoc 3."* ]]; then
  echo "Generating protobuf requires protoc 3.0.0-beta1 or newer. Please download and"
  echo "install the platform appropriate Protobuf package for your OS: "
  echo
  echo "  https://github.com/google/protobuf/releases"
  echo
  echo "WARNING: Protobuf changes are not being validated"
  exit 1
fi

gotoprotobuf=$(kube::util::find-binary "go-to-protobuf")

PACKAGES=(
  k8s.io/apiserver/pkg/apis/example/v1
  k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1
  k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1
  k8s.io/api/core/v1
  k8s.io/api/policy/v1beta1
  k8s.io/api/extensions/v1beta1
  k8s.io/api/autoscaling/v1
  k8s.io/api/authorization/v1
  k8s.io/api/autoscaling/v2beta1
  k8s.io/api/authorization/v1beta1
  k8s.io/api/batch/v1
  k8s.io/api/batch/v1beta1
  k8s.io/api/batch/v2alpha1
  k8s.io/api/apps/v1beta1
  k8s.io/api/apps/v1beta2
  k8s.io/api/apps/v1
  k8s.io/api/authentication/v1
  k8s.io/api/authentication/v1beta1
  k8s.io/api/rbac/v1alpha1
  k8s.io/api/rbac/v1beta1
  k8s.io/api/rbac/v1
  k8s.io/api/certificates/v1beta1
  k8s.io/api/imagepolicy/v1alpha1
  k8s.io/api/scheduling/v1alpha1
  k8s.io/api/settings/v1alpha1
  k8s.io/api/storage/v1beta1
  k8s.io/api/storage/v1
  k8s.io/api/admissionregistration/v1alpha1
  k8s.io/api/admission/v1alpha1
  k8s.io/api/networking/v1
  k8s.io/metrics/pkg/apis/metrics/v1alpha1
  k8s.io/metrics/pkg/apis/metrics/v1beta1
  k8s.io/metrics/pkg/apis/custom_metrics/v1beta1
  k8s.io/apiserver/pkg/apis/audit/v1alpha1
  k8s.io/apiserver/pkg/apis/audit/v1beta1
)

# requires the 'proto' tag to build (will remove when ready)
# searches for the protoc-gen-gogo extension in the output directory
# satisfies import of github.com/gogo/protobuf/gogoproto/gogo.proto and the
# core Google protobuf types
PATH="${KUBE_ROOT}/_output/bin:${PATH}" \
  "${gotoprotobuf}" \
  --proto-import="${KUBE_ROOT}/vendor" \
  --proto-import="${KUBE_ROOT}/third_party/protobuf" \
  --packages=$(IFS=, ; echo "${PACKAGES[*]}")
  "$@"
