#!/bin/bash -e
#
# This script fetches and rebuilds the "well-known types" protocol buffers.
# To run this you will need protoc and goprotobuf installed;
# see https://github.com/golang/protobuf for instructions.
# You also need Go and Git installed.

PKG=github.com/golang/protobuf/ptypes
UPSTREAM=https://github.com/google/protobuf
UPSTREAM_SUBDIR=src/google/protobuf
PROTO_FILES=(any duration empty struct timestamp wrappers)

function die() {
  echo 1>&2 $*
  exit 1
}

# Sanity check that the right tools are accessible.
for tool in go git protoc protoc-gen-go; do
  q=$(which $tool) || die "didn't find $tool"
  echo 1>&2 "$tool: $q"
done

tmpdir=$(mktemp -d -t regen-wkt.XXXXXX)
trap 'rm -rf $tmpdir' EXIT

echo -n 1>&2 "finding package dir... "
pkgdir=$(go list -f '{{.Dir}}' $PKG)
echo 1>&2 $pkgdir
base=$(echo $pkgdir | sed "s,/$PKG\$,,")
echo 1>&2 "base: $base"
cd "$base"

echo 1>&2 "fetching latest protos... "
git clone -q $UPSTREAM $tmpdir

for file in ${PROTO_FILES[@]}; do
  echo 1>&2 "* $file"
  protoc --go_out=. -I$tmpdir/src $tmpdir/src/google/protobuf/$file.proto || die
  cp $tmpdir/src/google/protobuf/$file.proto $PKG/$file
done

echo 1>&2 "All OK"
