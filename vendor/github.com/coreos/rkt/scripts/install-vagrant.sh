#!/bin/bash
set -e

cd /vagrant

echo "Installing common dependencies"
./scripts/install-common.sh
. /etc/profile

echo "Installing acbuild"
./scripts/install-acbuild.sh

echo "Building rkt"
./scripts/build-rkt.sh > rkt-build.log 2>&1

echo "Installing rkt"
./scripts/install-rkt.sh

groupadd rkt
./dist/scripts/setup-data-dir.sh
usermod -a -G rkt vagrant
