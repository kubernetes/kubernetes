#!/bin/bash

set -e
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev make automake wget git golang-go coreutils cpio squashfs-tools realpath autoconf file xz-utils patch bc locales libacl1-dev libssl-dev
