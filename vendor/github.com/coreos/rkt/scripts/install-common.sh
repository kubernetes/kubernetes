#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev make automake wget git coreutils cpio squashfs-tools realpath autoconf file libacl1-dev libtspi-dev bc

./scripts/install-go.sh
. /etc/profile

./scripts/install-appc-spec.sh
