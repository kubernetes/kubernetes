#!/bin/bash
# Copyright 2018 gRPC authors.
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
mkdir -p grpc/gcp
curl https://raw.githubusercontent.com/grpc/grpc-proto/master/grpc/gcp/altscontext.proto > grpc/gcp/altscontext.proto
curl https://raw.githubusercontent.com/grpc/grpc-proto/master/grpc/gcp/handshaker.proto > grpc/gcp/handshaker.proto
curl https://raw.githubusercontent.com/grpc/grpc-proto/master/grpc/gcp/transport_security_common.proto > grpc/gcp/transport_security_common.proto

protoc --go_out=plugins=grpc,paths=source_relative:. -I. grpc/gcp/*.proto
popd
rm -f proto/grpc_gcp/*.pb.go
cp "$TMP"/grpc/gcp/*.pb.go proto/grpc_gcp/

