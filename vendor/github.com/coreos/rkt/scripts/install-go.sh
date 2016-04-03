#!/bin/bash

set -e

export DEBIAN_FRONTEND=noninteractive
VERSION=1.5.3
OS=linux
ARCH=amd64

prefix=/usr/local

which wget || apt-get install -y wget
# not essential but go get depends on it
which git || apt-get install -y git

# grab go
if ! [ -e "$prefix/go" ]; then
    if ! [ -e "go$VERSION.$OS-$ARCH.tar.gz" ]; then
        wget -q https://storage.googleapis.com/golang/go$VERSION.$OS-$ARCH.tar.gz
    fi
    tar -C $prefix -xzf go$VERSION.$OS-$ARCH.tar.gz
fi

# setup user environment variables
if ! [ -e "/etc/profile.d/01go.sh" ]; then
    echo "export GOROOT=$prefix/go" | tee /etc/profile.d/01go.sh

    cat << 'EOF' | tee -a /etc/profile.d/go.sh

export GOPATH=$HOME/.gopath

[ -e $GOPATH ] || mkdir -p $GOPATH

export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
EOF

fi
