#!/bin/bash -e
#
# This script rebuilds the generated code for the protocol buffers.
# To run this you will need protoc and goprotobuf installed;
# see https://github.com/golang/protobuf for instructions.
# You also need Go and Git installed.

PKG=google.golang.org/cloud/bigtable
UPSTREAM=https://github.com/GoogleCloudPlatform/cloud-bigtable-client
UPSTREAM_SUBDIR=bigtable-protos/src/main/proto

function die() {
  echo 1>&2 $*
  exit 1
}

# Sanity check that the right tools are accessible.
for tool in go git protoc protoc-gen-go; do
  q=$(which $tool) || die "didn't find $tool"
  echo 1>&2 "$tool: $q"
done

tmpdir=$(mktemp -d -t regen-cbt.XXXXXX)
trap 'rm -rf $tmpdir' EXIT

echo -n 1>&2 "finding package dir... "
pkgdir=$(go list -f '{{.Dir}}' $PKG)
echo 1>&2 $pkgdir
base=$(echo $pkgdir | sed "s,/$PKG\$,,")
echo 1>&2 "base: $base"
cd $base

echo 1>&2 "fetching latest protos... "
git clone -q $UPSTREAM $tmpdir
# Pass 1: build mapping from upstream filename to our filename.
declare -A filename_map
for f in $(cd $PKG && find internal -name '*.proto'); do
  echo -n 1>&2 "looking for latest version of $f... "
  up=$(cd $tmpdir/$UPSTREAM_SUBDIR && find * -name $(basename $f))
  echo 1>&2 $up
  if [ $(echo $up | wc -w) != "1" ]; then
    die "not exactly one match"
  fi
  filename_map[$up]=$f
done
# Pass 2: build sed script for fixing imports.
import_fixes=$tmpdir/fix_imports.sed
for up in "${!filename_map[@]}"; do
  f=${filename_map[$up]}
  echo >>$import_fixes "s,\"$up\",\"$PKG/$f\","
done
cat $import_fixes | sed 's,^,### ,' 1>&2
# Pass 3: copy files, making necessary adjustments.
for up in "${!filename_map[@]}"; do
  f=${filename_map[$up]}
  cat $tmpdir/$UPSTREAM_SUBDIR/$up |
    # Adjust proto imports.
    sed -f $import_fixes |
    # Drop the UndeleteCluster RPC method. It returns a google.longrunning.Operation.
    sed '/^  rpc UndeleteCluster(/,/^  }$/d' |
    # Drop annotations, long-running operations and timestamps. They aren't supported (yet).
    sed '/"google\/longrunning\/operations.proto"/d' |
    sed '/google.longrunning.Operation/d' |
    sed '/"google\/protobuf\/timestamp.proto"/d' |
    sed '/google\.protobuf\.Timestamp/d' |
    sed '/"google\/api\/annotations.proto"/d' |
    sed '/option.*google\.api\.http.*{.*};$/d' |
    cat > $PKG/$f
done

# Run protoc once per package.
for dir in $(find $PKG/internal -name '*.proto' | xargs dirname | sort | uniq); do
  echo 1>&2 "* $dir"
  protoc --go_out=plugins=grpc:. $dir/*.proto
done
echo 1>&2 "All OK"
