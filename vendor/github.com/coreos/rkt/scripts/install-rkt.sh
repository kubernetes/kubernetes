#!/bin/bash

set -e

BUILDDIR=${BUILDDIR:-$PWD/build-rkt*}
sudo cp -v $BUILDDIR/bin/* /usr/local/bin
