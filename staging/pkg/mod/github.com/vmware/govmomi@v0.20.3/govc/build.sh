#!/bin/bash -e

git_version=$(git describe --dirty)
 if [[ $git_version == *-dirty ]] ; then
  echo 'Working tree is dirty.'
  echo 'NOTE: This script is meant for building govc releases via release.sh'
  echo 'To build govc from source see: https://github.com/vmware/govmomi/blob/master/govc/README.md#source'
  exit 1
fi

PROGRAM_NAME=govc
PROJECT_PKG="github.com/vmware/govmomi"
PROGRAM_PKG="${PROJECT_PKG}/${PROGRAM_NAME}"

export LDFLAGS="-w -X ${PROGRAM_PKG}/version.gitVersion=${git_version}"
export BUILD_OS="${BUILD_OS:-darwin linux windows freebsd}"
export BUILD_ARCH="${BUILD_ARCH:-386 amd64}"

set -x
make -C "$(go env GOPATH)/src/${PROGRAM_PKG}" -j build-all
set +x
