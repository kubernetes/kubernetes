#!/usr/bin/env bash
set -e

ORG_PATH="github.com/containernetworking"
REPO_PATH="${ORG_PATH}/cni"

if [ ! -h gopath/src/${REPO_PATH} ]; then
	mkdir -p gopath/src/${ORG_PATH}
	ln -s ../../../.. gopath/src/${REPO_PATH} || exit 255
fi

export GO15VENDOREXPERIMENT=1
export GOPATH=${PWD}/gopath

echo "Building API"
go build "$@" ${REPO_PATH}/libcni

echo "Building reference CLI"
go build -o ${PWD}/bin/cnitool "$@" ${REPO_PATH}/cnitool

echo "Building plugins"
PLUGINS="plugins/test/*"
for d in $PLUGINS; do
	if [ -d $d ]; then
		plugin=$(basename $d)
		echo "  " $plugin
		go build -o ${PWD}/bin/$plugin "$@" ${REPO_PATH}/$d
	fi
done
