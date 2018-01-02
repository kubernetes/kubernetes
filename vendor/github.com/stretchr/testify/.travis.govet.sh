#!/bin/bash

cd "$(dirname $0)"
DIRS=". assert require mock _codegen"
set -e
for subdir in $DIRS; do
  pushd $subdir
  go vet
  popd
done
