#!/usr/bin/env bash

set -ex

die() {
    echo "$@" >&2
    exit 1
}

cd ${HOME}

case "$PROTOBUF_VERSION" in
2*)
    basename=protobuf-$PROTOBUF_VERSION
    wget https://github.com/google/protobuf/releases/download/v$PROTOBUF_VERSION/$basename.tar.gz
    tar xzf $basename.tar.gz
    cd protobuf-$PROTOBUF_VERSION
    ./configure --prefix=${HOME} && make -j2 && make install
    ;;
3*)
    basename=protoc-$PROTOBUF_VERSION-linux-x86_64
    wget https://github.com/google/protobuf/releases/download/v$PROTOBUF_VERSION/$basename.zip
    unzip $basename.zip
    ;;
*)
    die "unknown protobuf version: $PROTOBUF_VERSION"
    ;;
esac
