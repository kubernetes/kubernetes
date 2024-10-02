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

set -eu -o pipefail

WORKDIR=$(mktemp -d)

function finish {
  rm -rf "$WORKDIR"
}
trap finish EXIT

export GOBIN=${WORKDIR}/bin
export PATH=${GOBIN}:${PATH}
mkdir -p ${GOBIN}

echo "remove existing generated files"
# grpc_testing_not_regenerate/*.pb.go is not re-generated,
# see grpc_testing_not_regenerate/README.md for details.
rm -f $(find . -name '*.pb.go' | grep -v 'grpc_testing_not_regenerate')

echo "go install google.golang.org/protobuf/cmd/protoc-gen-go"
(cd test/tools && go install google.golang.org/protobuf/cmd/protoc-gen-go)

echo "go install cmd/protoc-gen-go-grpc"
(cd cmd/protoc-gen-go-grpc && go install .)

echo "git clone https://github.com/grpc/grpc-proto"
git clone --quiet https://github.com/grpc/grpc-proto ${WORKDIR}/grpc-proto

echo "git clone https://github.com/protocolbuffers/protobuf"
git clone --quiet https://github.com/protocolbuffers/protobuf ${WORKDIR}/protobuf

# Pull in code.proto as a proto dependency
mkdir -p ${WORKDIR}/googleapis/google/rpc
echo "curl https://raw.githubusercontent.com/googleapis/googleapis/master/google/rpc/code.proto"
curl --silent https://raw.githubusercontent.com/googleapis/googleapis/master/google/rpc/code.proto > ${WORKDIR}/googleapis/google/rpc/code.proto

mkdir -p ${WORKDIR}/out

# Generates sources without the embed requirement
LEGACY_SOURCES=(
  ${WORKDIR}/grpc-proto/grpc/binlog/v1/binarylog.proto
  ${WORKDIR}/grpc-proto/grpc/channelz/v1/channelz.proto
  ${WORKDIR}/grpc-proto/grpc/health/v1/health.proto
  ${WORKDIR}/grpc-proto/grpc/lb/v1/load_balancer.proto
  profiling/proto/service.proto
  ${WORKDIR}/grpc-proto/grpc/reflection/v1alpha/reflection.proto
  ${WORKDIR}/grpc-proto/grpc/reflection/v1/reflection.proto
)

# Generates only the new gRPC Service symbols
SOURCES=(
  $(git ls-files --exclude-standard --cached --others "*.proto" | grep -v '^profiling/proto/service.proto$')
  ${WORKDIR}/grpc-proto/grpc/gcp/altscontext.proto
  ${WORKDIR}/grpc-proto/grpc/gcp/handshaker.proto
  ${WORKDIR}/grpc-proto/grpc/gcp/transport_security_common.proto
  ${WORKDIR}/grpc-proto/grpc/lookup/v1/rls.proto
  ${WORKDIR}/grpc-proto/grpc/lookup/v1/rls_config.proto
  ${WORKDIR}/grpc-proto/grpc/testing/*.proto
  ${WORKDIR}/grpc-proto/grpc/core/*.proto
)

# These options of the form 'Mfoo.proto=bar' instruct the codegen to use an
# import path of 'bar' in the generated code when 'foo.proto' is imported in
# one of the sources.
#
# Note that the protos listed here are all for testing purposes. All protos to
# be used externally should have a go_package option (and they don't need to be
# listed here).
OPTS=Mgrpc/core/stats.proto=google.golang.org/grpc/interop/grpc_testing/core,\
Mgrpc/testing/benchmark_service.proto=google.golang.org/grpc/interop/grpc_testing,\
Mgrpc/testing/stats.proto=google.golang.org/grpc/interop/grpc_testing,\
Mgrpc/testing/report_qps_scenario_service.proto=google.golang.org/grpc/interop/grpc_testing,\
Mgrpc/testing/messages.proto=google.golang.org/grpc/interop/grpc_testing,\
Mgrpc/testing/worker_service.proto=google.golang.org/grpc/interop/grpc_testing,\
Mgrpc/testing/control.proto=google.golang.org/grpc/interop/grpc_testing,\
Mgrpc/testing/test.proto=google.golang.org/grpc/interop/grpc_testing,\
Mgrpc/testing/payloads.proto=google.golang.org/grpc/interop/grpc_testing,\
Mgrpc/testing/empty.proto=google.golang.org/grpc/interop/grpc_testing

for src in ${SOURCES[@]}; do
  echo "protoc ${src}"
  protoc --go_out=${OPTS}:${WORKDIR}/out --go-grpc_out=${OPTS},use_generic_streams_experimental=true:${WORKDIR}/out \
    -I"." \
    -I${WORKDIR}/grpc-proto \
    -I${WORKDIR}/googleapis \
    -I${WORKDIR}/protobuf/src \
    ${src}
done

for src in ${LEGACY_SOURCES[@]}; do
  echo "protoc ${src}"
  protoc --go_out=${OPTS}:${WORKDIR}/out --go-grpc_out=${OPTS},require_unimplemented_servers=false:${WORKDIR}/out \
    -I"." \
    -I${WORKDIR}/grpc-proto \
    -I${WORKDIR}/googleapis \
    -I${WORKDIR}/protobuf/src \
    ${src}
done

# The go_package option in grpc/lookup/v1/rls.proto doesn't match the
# current location. Move it into the right place.
mkdir -p ${WORKDIR}/out/google.golang.org/grpc/internal/proto/grpc_lookup_v1
mv ${WORKDIR}/out/google.golang.org/grpc/lookup/grpc_lookup_v1/* ${WORKDIR}/out/google.golang.org/grpc/internal/proto/grpc_lookup_v1

# grpc_testing_not_regenerate/*.pb.go are not re-generated,
# see grpc_testing_not_regenerate/README.md for details.
rm ${WORKDIR}/out/google.golang.org/grpc/reflection/test/grpc_testing_not_regenerate/*.pb.go

cp -R ${WORKDIR}/out/google.golang.org/grpc/* .
