#!/bin/bash

set -e

./autogen.sh && ./configure --enable-tpm=no --with-stage1-default-images-directory=/usr/lib/rkt/stage1-images --with-stage1-default-location=/usr/lib/rkt/stage1-images/stage1-coreos.aci && make -j4
