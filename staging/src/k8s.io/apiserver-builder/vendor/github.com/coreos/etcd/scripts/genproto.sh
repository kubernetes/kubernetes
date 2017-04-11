#!/usr/bin/env bash
#
# Generate all etcd protobuf bindings.
# Run from repository root.
#
set -e

if ! [[ "$0" =~ "scripts/genproto.sh" ]]; then
	echo "must be run from repository root"
	exit 255
fi

# for now, be conservative about what version of protoc we expect
if ! [[ $(protoc --version) =~ "3.0.0" ]]; then
	echo "could not find protoc 3.0.0, is it installed + in PATH?"
	exit 255
fi

# directories containing protos to be built
DIRS="./wal/walpb ./etcdserver/etcdserverpb ./snap/snappb ./raft/raftpb ./mvcc/mvccpb ./lease/leasepb ./auth/authpb"

# exact version of protoc-gen-gogo to build
GOGO_PROTO_SHA="5f813990bfffa3c2f4414dbea480e705ab280358"
GRPC_GATEWAY_SHA="c8ec92d0481dd77d9b8c1808eb6476d190aa039a"

# set up self-contained GOPATH for building
export GOPATH=${PWD}/gopath
export GOBIN=${PWD}/bin
export PATH="${GOBIN}:${PATH}"

COREOS_ROOT="${GOPATH}/src/github.com/coreos"
ETCD_ROOT="${COREOS_ROOT}/etcd"
GOGOPROTO_ROOT="${GOPATH}/src/github.com/gogo/protobuf"
GOGOPROTO_PATH="${GOGOPROTO_ROOT}:${GOGOPROTO_ROOT}/protobuf"
GRPC_GATEWAY_ROOT="${GOPATH}/src/github.com/grpc-ecosystem/grpc-gateway"

rm -f "${ETCD_ROOT}"
mkdir -p "${COREOS_ROOT}"
ln -s "${PWD}" "${ETCD_ROOT}"

# Ensure we have the right version of protoc-gen-gogo by building it every time.
# TODO(jonboulle): vendor this instead of `go get`ting it.
go get -u github.com/gogo/protobuf/{proto,protoc-gen-gogo,gogoproto}
go get -u golang.org/x/tools/cmd/goimports
pushd "${GOGOPROTO_ROOT}"
	git reset --hard "${GOGO_PROTO_SHA}"
	make install
popd

# generate gateway code
go get -u github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway
go get -u github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger
pushd "${GRPC_GATEWAY_ROOT}"
	git reset --hard "${GRPC_GATEWAY_SHA}"
	go install ./protoc-gen-grpc-gateway
popd

for dir in ${DIRS}; do
	pushd ${dir}
		protoc --gofast_out=plugins=grpc,import_prefix=github.com/coreos/:. -I=.:"${GOGOPROTO_PATH}":"${COREOS_ROOT}":"${GRPC_GATEWAY_ROOT}/third_party/googleapis" *.proto
		sed -i.bak -E "s/github\.com\/coreos\/(gogoproto|github\.com|golang\.org|google\.golang\.org)/\1/g" *.pb.go
		sed -i.bak -E 's/github\.com\/coreos\/(errors|fmt|io)/\1/g' *.pb.go
		sed -i.bak -E 's/import _ \"gogoproto\"//g' *.pb.go
		sed -i.bak -E 's/import fmt \"fmt\"//g' *.pb.go
		sed -i.bak -E 's/import _ \"github\.com\/coreos\/google\/api\"//g' *.pb.go
		rm -f *.bak
		goimports -w *.pb.go
	popd
done

protoc -I. \
    -I${GRPC_GATEWAY_ROOT}/third_party/googleapis \
    -I${GOGOPROTO_PATH} \
    -I${COREOS_ROOT} \
    --grpc-gateway_out=logtostderr=true:. \
    --swagger_out=logtostderr=true:./Documentation/dev-guide/apispec/swagger/. \
    ./etcdserver/etcdserverpb/rpc.proto

# TODO: change this whenever we add more swagger API
mv \
	Documentation/dev-guide/apispec/swagger/etcdserver/etcdserverpb/rpc.swagger.json \
	Documentation/dev-guide/apispec/swagger/rpc.swagger.json
rm -rf Documentation/dev-guide/apispec/swagger/etcdserver/etcdserverpb


# install protodoc
# go get -v -u github.com/coreos/protodoc
#
# by default, do not run this option.
# only run when './scripts/genproto.sh -g'
#
if [ "$1" = "-g" ]; then
	echo "protodoc is auto-generating grpc API reference documentation..."
	go get -v -u github.com/coreos/protodoc
	SHA_PROTODOC="f4164b1cce80b5eba4c835d08483f552dc568b7c"
	PROTODOC_PATH="${GOPATH}/src/github.com/coreos/protodoc"
	pushd "${PROTODOC_PATH}"
		git reset --hard "${SHA_PROTODOC}"
		go install
		echo "protodoc is updated"
	popd

	protodoc --directories="etcdserver/etcdserverpb=service_message,mvcc/mvccpb=service_message,lease/leasepb=service_message,auth/authpb=service_message" \
		--title="etcd API Reference" \
		--output="Documentation/dev-guide/api_reference_v3.md" \
		--message-only-from-this-file="etcdserver/etcdserverpb/rpc.proto" \
		--disclaimer="This is a generated documentation. Please read the proto files for more."

	echo "protodoc is finished..."
else
	echo "skipping grpc API reference document auto-generation..."
fi

