#!/bin/bash

set -xe

SRC_DIR="${SRC_DIR:-$PWD}"
BUILDDIR="${BUILDDIR:-$PWD/build-rir}"
RKT_BUILDER_ACI="${RKT_BUILDER_ACI:-coreos.com/rkt/builder:1.1.1}"
RKT_CMD="${RKT_CMD:-rkt}"

mkdir -p $BUILDDIR

$RKT_CMD run \
    --dns=8.8.8.8 \
    --volume src-dir,kind=host,source="${SRC_DIR}" \
    --volume build-dir,kind=host,source="${BUILDDIR}" \
    --interactive \
    $RKT_BUILDER_ACI $@
