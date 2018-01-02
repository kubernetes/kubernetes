#!/bin/sh

# This is needed to build the executable on an x86 machine for RPi

fail() {
    echo "$1"
    exit 1
}

DOCKERFILEDIR=`pwd`
compile() {
    cd ../../..
    env GOOS=linux GOARCH=arm make || fail "Unable to create build"
    cp heketi $DOCKERFILEDIR
    cp client/cli/go/heketi-cli $DOCKERFILEDIR
    make clean
    cd $DOCKERFILEDIR
}

build() {
    sudo docker build --rm --tag heketi/heketi-rpi:latest .
}

compile
build
