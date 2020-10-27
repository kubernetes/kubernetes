#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# This script builds protoc-gen-gogo binary in runtime and genertates 
# `*/api.pb.go` from the protobuf file `*/api.proto`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
KUBE_REMOTE_RUNTIME_ROOT="${KUBE_ROOT}/staging/src/k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

go install k8s.io/kubernetes/vendor/k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo

if [[ -z "$(which protoc)" || "$(protoc --version)" != "libprotoc 3."* ]]; then
  echo "Generating protobuf requires protoc 3.0.0-beta1 or newer. Please download and"
  echo "install the platform appropriate Protobuf package for your OS: "
  echo
  echo "  https://github.com/google/protobuf/releases"
  echo
  echo "WARNING: Protobuf changes are not being validated"
  exit 1
fi

function cleanup {
	rm -f "${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go.bak"
	rm -f "${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go.tmp"
}

trap cleanup EXIT

gogopath=$(dirname "$(kube::util::find-binary "protoc-gen-gogo")")

PATH="${gogopath}:${PATH}" \
  protoc \
  --proto_path="${KUBE_REMOTE_RUNTIME_ROOT}" \
  --proto_path="${KUBE_ROOT}/vendor" \
  --gogo_out=plugins=grpc:"${KUBE_REMOTE_RUNTIME_ROOT}" "${KUBE_REMOTE_RUNTIME_ROOT}/api.proto"

# Update boilerplate for the generated file.
cat hack/boilerplate/boilerplate.generatego.txt "${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go" > "${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go.tmp"
mv "${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go.tmp" "${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go"

# Run gofmt to clean up the generated code.
kube::golang::verify_go_version
gofmt -l -s -w "${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go"
