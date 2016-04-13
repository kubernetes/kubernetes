#!/usr/bin/env bash

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

set -ex

function check_tool {
if ! which $1; then
    echo "Get $1 and put it in your \$PATH" >&2;
    exit 1;
fi
}

MODIFY=${MODIFY:-""}
FLAGS=${FLAGS:-""}
IMG=${IMG:-"debian"}
IMG_VERSION=${IMG_VERSION:-"sid"}
DOCKERIMG="$IMG:$IMG_VERSION"
ACI_FILE=${ACI_FILE:-"./library-$IMG-$IMG_VERSION.aci"}
OUT_ACI=${OUT_ACI:-"rkt-builder.aci"}
ACI_NAME=${ACI_NAME:-"coreos.com/rkt/builder"}
BUILDDIR=/opt/build-rkt
SRC_DIR=/opt/rkt
ACI_GOPATH=/go
VERSION=${VERSION:-"v1.2.1+git"}
echo "Version: $VERSION"

echo "Building $ACI_FILE"

check_tool acbuild
check_tool docker2aci
# check_tool actool

if [ ! -f "$ACI_FILE" ]; then
    docker2aci "docker://$DOCKERIMG"
    # These base images don't always come with valid values
    # actool patch-manifest -user 0 -group 0 --name $IMG-$IMG_VERSION --exec /bin/bash --replace $ACI_FILE
fi

acbuildend () {
    export EXIT=$?;
    acbuild --debug end && exit $EXIT;
}

# If modify is specified, pass the modify flag to each command and don't use
# acbuild begin, write or end otherwise build with a context, and setup a trap
# to handle failures,
if [ "$MODIFY" ]; then
    FLAGS="--modify $MODIFY $FLAGS"
    OUT_ACI=$MODIFY
else
    acbuild $FLAGS begin "$ACI_FILE"
    trap acbuildend EXIT
fi

acbuild $FLAGS set-name $ACI_NAME
acbuild $FLAGS label add version $VERSION
acbuild $FLAGS set-user 0
acbuild $FLAGS set-group 0
acbuild $FLAGS environment add OS_VERSION $IMG_VERSION
acbuild $FLAGS environment add GOPATH $ACI_GOPATH
acbuild $FLAGS environment add BUILDDIR $BUILDDIR
acbuild $FLAGS environment add SRC_DIR $SRC_DIR
acbuild $FLAGS mount add build-dir $BUILDDIR
acbuild $FLAGS mount add src-dir $SRC_DIR
acbuild $FLAGS set-working-dir $SRC_DIR
acbuild $FLAGS copy "$(dirname $0)" /scripts
acbuild $FLAGS run /bin/bash "/scripts/install-deps-$IMG-$IMG_VERSION.sh"
acbuild $FLAGS run /bin/bash /scripts/install-appc-spec.sh
acbuild $FLAGS set-exec /bin/bash /scripts/build-rkt.sh
if [ -z "$MODIFY" ]; then
    acbuild $FLAGS write --overwrite $OUT_ACI
fi
