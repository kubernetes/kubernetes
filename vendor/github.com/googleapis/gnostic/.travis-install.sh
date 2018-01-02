#!/bin/sh

#
# Install dependencies that aren't available as Ubuntu packages.
#
# Everything goes into $HOME/local. 
#
# Scripts should add 
# - $HOME/local/bin to PATH 
# - $HOME/local/lib to LD_LIBRARY_PATH
#

cd
mkdir -p local

# Install swift
SWIFT_BRANCH=swift-3.0.2-release
SWIFT_VERSION=swift-3.0.2-RELEASE
SWIFT_PLATFORM=ubuntu14.04
SWIFT_URL=https://swift.org/builds/$SWIFT_BRANCH/$(echo "$SWIFT_PLATFORM" | tr -d .)/$SWIFT_VERSION/$SWIFT_VERSION-$SWIFT_PLATFORM.tar.gz

echo $SWIFT_URL

curl -fSsL $SWIFT_URL -o swift.tar.gz 
tar -xzf swift.tar.gz --strip-components=2 --directory=local

# Install protoc
PROTOC_URL=https://github.com/google/protobuf/releases/download/v3.2.0rc2/protoc-3.2.0rc2-linux-x86_64.zip

echo $PROTOC_URL

curl -fSsL $PROTOC_URL -o protoc.zip
unzip protoc.zip -d local

# Verify installation
find local
