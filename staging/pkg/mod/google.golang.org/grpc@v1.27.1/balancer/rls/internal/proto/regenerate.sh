#!/bin/bash
# Copyright 2020 gRPC authors.
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
mkdir -p grpc/rls/grpc_lookup_v1
curl https://raw.githubusercontent.com/grpc/grpc-proto/master/grpc/lookup/v1/rls.proto > grpc/rls/grpc_lookup_v1/rls.proto
curl https://raw.githubusercontent.com/grpc/grpc-proto/master/grpc/lookup/v1/rls_config.proto > grpc/rls/grpc_lookup_v1/rls_config.proto

protoc --go_out=plugins=grpc,paths=source_relative:. -I. grpc/rls/grpc_lookup_v1/*.proto
popd
rm -f ./grpc_lookup_v1/*.pb.go
cp "$TMP"/grpc/rls/grpc_lookup_v1/*.pb.go ../../../rls/internal/proto/grpc_lookup_v1/
