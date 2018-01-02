#!/bin/bash
set -e

if [ -z "$WORKSPACE" ]
then
  pushd $(dirname $BASH_SOURCE)
  export WORKSPACE=$(git rev-parse --show-toplevel)
  echo Assume default WORKSPACE $WORKSPACE
  popd
fi

# Go expects to have a "Go workspace" (GOPATH) present in order to build.
# This workspace is not supposed to be checked in, so we must create it at
# build time.
cd $WORKSPACE
export GOPATH=$WORKSPACE

# $WORKSPACE will be the root of the git repo that is pulled in by Jenkins.
# We need to move its contents into the expected package path inside
# $GOPATH/src (defined by PACKAGESRC) before we can build.
PACKAGESRC=src/github.com/vmware/photon-controller-go-sdk

# clean the directory
if [ -n "$WORKSPACE" ]
then
	rm -rf $WORKSPACE/src $WORKSPACE/pkg $WORKSPACE/bin
fi

REPOFILES=(*)
echo ${REPOFILES[*]}
mkdir -p $PACKAGESRC
cp -r ${REPOFILES[*]} $PACKAGESRC/

go get github.com/tools/godep
pushd $PACKAGESRC
$GOPATH/bin/godep restore

# Fail if there is any go fmt error.
if [[ -n "$(gofmt -l photon)" ]]; then
	echo Fix gofmt errors
	gofmt -d photon
	exit 1
fi

export cmd="go test ./... -v"
# Test against a real external endpoint requires longer timeout than GO's default 600s
if [ "" != "$TEST_ENDPOINT" ]
	then
		export cmd="$cmd -timeout 1800s"
fi
# Build and run tests
$cmd -ginkgo.noColor -ginkgo.slowSpecThreshold=60 -ginkgo.v
