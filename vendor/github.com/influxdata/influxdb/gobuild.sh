#!/bin/bash
# This script run inside the Dockerfile_build_ubuntu64_git container and
# gets the latests Go source code and compiles it.
# Then passes control over to the normal build.py script

set -e

cd /go/src
git fetch --all
git checkout $GO_CHECKOUT
# Merge in recent changes if we are on a branch
# if we checked out a tag just ignore the error
git pull || true
./make.bash

# Run normal build.py
cd "$PROJECT_DIR"
exec ./build.py "$@"
