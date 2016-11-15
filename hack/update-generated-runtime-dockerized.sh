#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
KUBE_REMOTE_RUNTIME_ROOT="${KUBE_ROOT}/pkg/kubelet/api/v1alpha1/runtime"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

BINS=(
	cmd/libs/go2idl/go-to-protobuf/protoc-gen-gogo
)
make -C "${KUBE_ROOT}" WHAT="${BINS[*]}"

if [[ -z "$(which protoc)" || "$(protoc --version)" != "libprotoc 3.0."* ]]; then
  echo "Generating protobuf requires protoc 3.0.0-beta1 or newer. Please download and"
  echo "install the platform appropriate Protobuf package for your OS: "
  echo
  echo "  https://github.com/google/protobuf/releases"
  echo
  echo "WARNING: Protobuf changes are not being validated"
  exit 1
fi

function cleanup {
	rm -f ${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go.bak
}

trap cleanup EXIT

gogopath=$(dirname $(kube::util::find-binary "protoc-gen-gogo"))
export PATH=$gogopath:$PATH
protoc -I${KUBE_REMOTE_RUNTIME_ROOT} --gogo_out=plugins=grpc:${KUBE_REMOTE_RUNTIME_ROOT} ${KUBE_REMOTE_RUNTIME_ROOT}/api.proto
echo "$(cat hack/boilerplate/boilerplate.go.txt ${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go)" > ${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go
sed -i".bak" "s/Copyright YEAR/Copyright $(date '+%Y')/g" ${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go

