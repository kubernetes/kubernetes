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
mkdir -p grpc/service_config
curl https://raw.githubusercontent.com/grpc/grpc-proto/master/grpc/service_config/service_config.proto > grpc/service_config/service_config.proto
mkdir -p google/rpc
curl https://raw.githubusercontent.com/googleapis/googleapis/master/google/rpc/code.proto > google/rpc/code.proto

protoc --go_out=plugins=grpc,paths=source_relative:. -I. grpc/service_config/*.proto
popd
rm -f ./*.pb.go
cp "$TMP"/grpc/service_config/*.pb.go ./

