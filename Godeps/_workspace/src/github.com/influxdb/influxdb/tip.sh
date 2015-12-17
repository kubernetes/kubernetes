#!/bin/bash
# This script run inside the Dockerfile_build_ubuntu64_tip container and
# gets the latests Go source code and compiles it.
# Then passes control over to the normal build.py script

set -e

cd /go/src
git pull
./make.bash

# Run normal build.py
cd "$PROJECT_DIR"
exec ./build.py "$@"
