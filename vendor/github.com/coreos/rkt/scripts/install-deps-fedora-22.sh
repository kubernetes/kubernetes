#!/bin/bash

set -e

FEDORA22_DEPS="make gcc glibc-devel glibc-static cpio squashfs-tools gpg autoconf automake golang file git wget tar xz patch bc hostname findutils openssl libacl-devel openssl-devel systemd-devel"

dnf install -y ${FEDORA22_DEPS}
