#!/bin/bash
#
# Generate the Go code for the generator and
# transformer factory functions in
#
#   sigs.k8s.io/kustomize/plugin/builtin
#
# from the raw plugin directories found _below_
# that directory.

set -e

myGoPath=$1
if [ -z ${1+x} ]; then
  myGoPath=$GOPATH
fi

if [ -z "$myGoPath" ]; then
  echo "Must specify a GOPATH"
  exit 1
fi

dir=$myGoPath/src/sigs.k8s.io/kustomize

if [ ! -d "$dir" ]; then
  echo "$dir is not a directory."
  exit 1
fi

echo Generating linkable plugins...

pushd $dir >& /dev/null

GOPATH=$myGoPath go generate \
    sigs.k8s.io/kustomize/plugin/builtin/...
GOPATH=$myGoPath go fmt \
    sigs.k8s.io/kustomize/plugin/builtin

popd >& /dev/null

echo All done.
