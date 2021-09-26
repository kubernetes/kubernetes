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

# Install protoc
PROTOC_URL=https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip
echo $PROTOC_URL
curl -fSsL $PROTOC_URL -o protoc.zip
unzip protoc.zip -d local

# Verify installation
find local
