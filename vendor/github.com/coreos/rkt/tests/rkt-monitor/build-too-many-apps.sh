#!/usr/bin/env bash
set -e

if ! [[ "$0" =~ "tests/rkt-monitor/build-too-many-apps.sh" ]]; then
	echo "must be run from repository root"
	exit 255
fi

NUM_APPS=100

# Build worker binary

echo "Building worker binary"
make rkt-monitor

# Generate worker images

acbuildEnd() {
    export EXIT=$?
    acbuild --debug end && exit $EXIT
}
trap acbuildEnd EXIT

mkdir -p ./too-many-apps-images

for i in $(seq 1 ${NUM_APPS});
do
    NAME="worker-${i}"

    echo "Building image ${NAME}"

    acbuild begin
    acbuild copy build-rkt-1.8.0+git/bin/sleeper /worker-binary
    acbuild set-exec /worker-binary
    acbuild set-name ${NAME}
    acbuild write --overwrite too-many-apps-images/${NAME}.aci
    acbuild end
done

# Generate pod manifest

echo "Generating pod manifest"

OUTPUT=${PWD}/too-many-apps.podmanifest

cat <<EOF >${OUTPUT}
{
    "acVersion": "0.8.10",
    "acKind": "PodManifest",
    "apps": [
EOF

appSection() {
    SHA512=$(gunzip too-many-apps-images/${1}.aci --stdout|sha512sum)
    SHA512="sha512-${SHA512:0:32}"

cat <<EOF >>$2
{
    "name": "$1",
    "image": {
        "name": "$1",
        "id": "$SHA512",
        "labels": [
            {
                "name":  "arch",
                "value": "amd64"
            },
            {
                "name":  "os",
                "value": "linux"
            }
        ]
    },
    "app": {
        "exec": [ "/worker-binary" ],
        "group": "0",
        "user": "0"
    }
}
EOF
}

appSection "worker-1" ${OUTPUT}
for i in $(seq 2 ${NUM_APPS});
do
    echo ',' >> ${OUTPUT}
    appSection "worker-${i}" ${OUTPUT}
done

echo ']}' >> ${OUTPUT}

