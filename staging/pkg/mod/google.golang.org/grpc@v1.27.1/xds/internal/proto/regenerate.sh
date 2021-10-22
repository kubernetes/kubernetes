#!/bin/bash
# Copyright 2019 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eux -o pipefail

TMP=$(mktemp -d)

function finish {
  rm -rf "$TMP"
}
trap finish EXIT

pushd "$TMP"

UDPA_VERSION=015fc86d90f4045a56f831bcdfa560bc455450e2
git clone https://github.com/cncf/udpa.git
cd udpa
git checkout ${UDPA_VERSION}

VALIDATE_VERSION=4f00761ef740eb579cfddcfee3951e13c4fae6f8
mkdir validate
curl https://raw.githubusercontent.com/envoyproxy/protoc-gen-validate/${VALIDATE_VERSION}/validate/validate.proto > validate/validate.proto

find udpa -name "*.proto" | xargs -L 1 protoc --go_out=plugins=grpc,paths=source_relative,Mudpa/data/orca/v1/orca_load_report.proto=google.golang.org/grpc/xds/internal/proto/udpa/data/orca/v1:.
popd
rm -rf ./udpa
cp -r "$TMP"/udpa/udpa ./udpa
find udpa -type f -not -name "*.pb.go" | xargs rm
