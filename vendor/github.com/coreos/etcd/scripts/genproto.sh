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

PREFIX="github.com/coreos/etcd/Godeps/_workspace/src"
ESCAPED_PREFIX=$(echo $PREFIX | sed -e 's/[\/&]/\\&/g')

# directories containing protos to be built
DIRS="./wal/walpb ./etcdserver/etcdserverpb ./snap/snappb ./raft/raftpb ./storage/storagepb ./lease/leasepb"

# exact version of protoc-gen-gogo to build
SHA="c57e439bad574c2e0877ff18d514badcfced004d"

# set up self-contained GOPATH for building
export GOPATH=${PWD}/gopath
export GOBIN=${PWD}/bin
export PATH="${GOBIN}:${PATH}"

COREOS_ROOT="${GOPATH}/src/github.com/coreos"
ETCD_ROOT="${COREOS_ROOT}/etcd"
GOGOPROTO_ROOT="${GOPATH}/src/github.com/gogo/protobuf"
GOGOPROTO_PATH="${GOGOPROTO_ROOT}:${GOGOPROTO_ROOT}/protobuf"

rm -f "${ETCD_ROOT}"
mkdir -p "${COREOS_ROOT}"
ln -s "${PWD}" "${ETCD_ROOT}"

# Ensure we have the right version of protoc-gen-gogo by building it every time.
# TODO(jonboulle): vendor this instead of `go get`ting it.
go get github.com/gogo/protobuf/{proto,protoc-gen-gogo,gogoproto}
go get golang.org/x/tools/cmd/goimports
pushd "${GOGOPROTO_ROOT}"
	git reset --hard "${SHA}"
	make install
popd

for dir in ${DIRS}; do
	pushd ${dir}
		protoc --gogofast_out=plugins=grpc,import_prefix=github.com/coreos/:. -I=.:"${GOGOPROTO_PATH}":"${COREOS_ROOT}" *.proto
		sed -i.bak -E "s/github\.com\/coreos\/(gogoproto|github\.com|golang\.org|google\.golang\.org)/${ESCAPED_PREFIX}\/\1/g" *.pb.go
		sed -i.bak -E 's/github\.com\/coreos\/(errors|fmt|io)/\1/g' *.pb.go
		sed -i.bak -E 's/import _ \"github\.com\/coreos\/etcd\/Godeps\/\_workspace\/src\/gogoproto\"//g' *.pb.go
		sed -i.bak -E 's/import fmt \"fmt\"//g' *.pb.go
		rm -f *.bak
		goimports -w *.pb.go
	popd
done
