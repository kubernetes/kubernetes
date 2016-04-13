#!/bin/bash

set -xe

SRC_DIR=${SRC_DIR:-$PWD}
BUILDDIR=${BUILDDIR:-$PWD/build-rir}
RKT_BUILDER_ACI=${RKT_BUILDER_ACI:-coreos.com/rkt/builder:v1.2.1+git}

mkdir -p $BUILDDIR

rkt run \
    --inherit-env \
    --volume src-dir,kind=host,source=$SRC_DIR \
    --volume build-dir,kind=host,source=$BUILDDIR \
    --interactive \
    $RKT_BUILDER_ACI $@
