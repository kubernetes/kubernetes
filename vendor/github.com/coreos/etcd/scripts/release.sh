#!/usr/bin/env bash
#
# Build all release binaries and images to directory ./release.
# Run from repository root.
#
set -e

VERSION=$1
if [ -z "${VERSION}" ]; then
	echo "Usage: ${0} VERSION" >> /dev/stderr
	exit 255
fi

if ! command -v acbuild >/dev/null; then
    echo "cannot find acbuild"
    exit 1
fi

if ! command -v docker >/dev/null; then
    echo "cannot find docker"
    exit 1
fi

ETCD_ROOT=$(dirname "${BASH_SOURCE}")/..

pushd ${ETCD_ROOT} >/dev/null
	echo Building etcd binary...
	./scripts/build-binary ${VERSION}
	echo Building aci image...
	BINARYDIR=release/etcd-${VERSION}-linux-amd64 BUILDDIR=release ./scripts/build-aci ${VERSION}
	echo Building docker image...
	BINARYDIR=release/etcd-${VERSION}-linux-amd64 BUILDDIR=release ./scripts/build-docker ${VERSION}
popd >/dev/null
