#!/usr/bin/env bash
set -e

# Gets the directory that this script is stored in.
# https://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ORG_PATH="github.com/appc"
REPO_PATH="${ORG_PATH}/spec"

if [ ! -h ${DIR}/gopath/src/${REPO_PATH} ]; then
  mkdir -p ${DIR}/gopath/src/${ORG_PATH}
  cd ${DIR} && ln -s ../../../.. gopath/src/${REPO_PATH} || exit 255
fi

export GO15VENDOREXPERIMENT=1
export GOBIN=${DIR}/bin
export GOPATH=${DIR}/gopath
export GOOS GOARCH

eval $(go env)

if [ "${GOOS}" = "freebsd" ]; then
  # /usr/bin/cc is clang on freebsd, but we need to tell it to go to
  # make it generate proper flavour of code that doesn't emit
  # warnings.
  export CC=clang
fi

echo "Building actool..."
go build -o ${GOBIN}/actool ${REPO_PATH}/actool

if ! [[ -d "$(go env GOROOT)/pkg/${GOOS}_${GOARCH}" ]]; then
	echo "go ${GOOS}/${GOARCH} not bootstrapped, not building ACE validator"
else
	echo "Building ACE validator..."
	CGO_ENABLED=0 go build -a -installsuffix ace -ldflags '-extldflags "-static"' -o ${GOBIN}/ace-validator ${REPO_PATH}/ace
fi
