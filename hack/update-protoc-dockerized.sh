#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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
KUBE_PACKAGE="$1"
source "${KUBE_ROOT}/hack/lib/init.sh"

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 PATH_TO_PACKAGE" >&2
	echo "Example: $0 ${KUBE_ROOT}/pkg/kubelet/apis/cri/v1alpha1/runtime/" >&2
	exit 1
fi

kube::golang::setup_env

BINS=(
	vendor/k8s.io/kube-gen/cmd/go-to-protobuf/protoc-gen-gogo
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

function cleanup {
	rm -f ${KUBE_PACKAGE}/api.pb.go.bak
}

trap cleanup EXIT

gogopath=$(dirname $(kube::util::find-binary "protoc-gen-gogo"))

PATH="${gogopath}:${PATH}" \
  protoc \
  --proto_path="${KUBE_PACKAGE}" \
  --proto_path="${KUBE_ROOT}/vendor" \
  --gogo_out=plugins=grpc:${KUBE_PACKAGE} ${KUBE_PACKAGE}/api.proto

# Update boilerplate for the generated file.
echo "$(cat hack/boilerplate/boilerplate.go.txt ${KUBE_PACKAGE}/api.pb.go)" \
  > ${KUBE_PACKAGE}/api.pb.go

sed -i".bak" "s/Copyright YEAR/Copyright $(date '+%Y')/g" ${KUBE_PACKAGE}/api.pb.go

# Run gofmt to clean up the generated code.
kube::golang::verify_go_version
gofmt -l -s -w ${KUBE_PACKAGE}/api.pb.go
