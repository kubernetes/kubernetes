#!/bin/bash
#
# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script rebuilds the generated code for the protocol buffers.
# To run this you will need protoc and goprotobuf installed;
# see https://github.com/golang/protobuf for instructions.
# You also need Go and Git installed.

set -e

PKG=google.golang.org/genproto
PROTO_REPO=https://github.com/google/protobuf
PROTO_SUBDIR=src/google/protobuf
API_REPO=https://github.com/googleapis/googleapis

function die() {
  echo 1>&2 $*
  exit 1
}

# Sanity check that the right tools are accessible.
for tool in go git protoc protoc-gen-go; do
  q=$(which $tool) || die "didn't find $tool"
  echo 1>&2 "$tool: $q"
done

root=$(go list -f '{{.Root}}' $PKG/... | head -n1)
if [ -z "$root" ]; then
  die "cannot find root of $PKG"
fi

remove_dirs=
trap 'rm -rf $remove_dirs' EXIT

if [ -z "$PROTOBUF" ]; then
  proto_repo_dir=$(mktemp -d -t regen-cds-proto.XXXXXX)
  git clone -q $PROTO_REPO $proto_repo_dir &
  remove_dirs="$proto_repo_dir"
  # The protoc include directory is actually the "src" directory of the repo.
  protodir="$proto_repo_dir/src"
else
  protodir="$PROTOBUF/src"
fi

if [ -z "$GOOGLEAPIS" ]; then
  apidir=$(mktemp -d -t regen-cds-api.XXXXXX)
  git clone -q $API_REPO $apidir &
  remove_dirs="$remove_dirs $apidir"
else
  apidir="$GOOGLEAPIS"
fi

wait

# Nuke everything, we'll generate them back
rm -r googleapis/ protobuf/

go run regen.go -go_out "$root/src" -pkg_prefix "$PKG" "$apidir" "$protodir"

# Sanity check the build.
echo 1>&2 "Checking that the libraries build..."
go build -v ./...

echo 1>&2 "All done!"
