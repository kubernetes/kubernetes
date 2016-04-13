#!/bin/bash
set -e

if ! [ -e "$PWD/acbuild" ]; then
    git clone https://github.com/appc/acbuild
fi

pushd acbuild

git pull
./build
cp -v bin/* /usr/local/bin

popd
