#!/bin/bash -e
#
# This script rebuilds the generated code for the protocol buffers.
# To run this you will need protoc and goprotobuf installed;
# see https://github.com/golang/protobuf for instructions.

PKG=google.golang.org/appengine

function die() {
	echo 1>&2 $*
	exit 1
}

# Sanity check that the right tools are accessible.
for tool in go protoc protoc-gen-go; do
	q=$(which $tool) || die "didn't find $tool"
	echo 1>&2 "$tool: $q"
done

echo -n 1>&2 "finding package dir... "
pkgdir=$(go list -f '{{.Dir}}' $PKG)
echo 1>&2 $pkgdir
base=$(echo $pkgdir | sed "s,/$PKG\$,,")
echo 1>&2 "base: $base"
cd $base
for f in $(find $PKG/internal -name '*.proto'); do
	echo 1>&2 "* $f"
	protoc --go_out=. $f
done

# Fix up import lines.
# This should be fixed upstream.
# https://code.google.com/p/goprotobuf/issues/detail?id=32
for f in $(find $PKG/internal -name '*.pb.go'); do
  sed -i '/^import.*\.pb"$/s,/[^/]*\.pb"$,",' $f
done
