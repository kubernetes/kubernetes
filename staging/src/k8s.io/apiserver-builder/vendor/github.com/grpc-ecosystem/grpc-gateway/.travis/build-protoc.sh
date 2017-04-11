#!/bin/sh -eu
protoc_version=$1
if test -z "${protoc_version}"; then
	echo "Usage: .travis/build-protoc.sh protoc-version"
	exit 1
fi
if ! $HOME/local/bin/protoc-${protoc_version} --version 2>/dev/null; then
	rm -rf $HOME/local

	mkdir -p $HOME/tmp
	cd $HOME/tmp
	wget https://github.com/google/protobuf/archive/v${protoc_version}.tar.gz
	tar xvzf v${protoc_version}.tar.gz
	cd protobuf-${protoc_version}
	./autogen.sh
	./configure --prefix=$HOME/local --program-suffix=-${protoc_version}
	make -j 4
	make install
fi
ln -sf $HOME/local/bin/protoc-${protoc_version} $HOME/local/bin/protoc

echo \$ $HOME/local/bin/protoc --version
$HOME/local/bin/protoc --version
