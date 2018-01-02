#!/bin/bash

set -e

DEBIAN_SID_DEPS="ca-certificates gcc libc6-dev gpg make automake wget git golang-go coreutils cpio squashfs-tools realpath autoconf file xz-utils patch bc locales libacl1-dev libssl-dev libtspi-dev libsystemd-dev"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ${DEBIAN_SID_DEPS}
