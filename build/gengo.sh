#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
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

if [[ -z "${STARTED_FROM_GENGO_BZL:-}" ]]; then
  echo "Call this script from build/gengo.bzl"
  exit 1
fi

# Main logic which runs the gengo tool on packages requesting generation.

# Set DEBUG=1 for extra info and checks
DEBUG=
# location of the prebuilt generator
dcg="$PWD/$tool"

# original pwd
O=$PWD

# split each GOPATH entry by :
gopaths=($(IFS=: ; echo $GOPATH))
# The first GOPATH entry is where go_genrule puts input source files.
# So use this one
srcpath="${gopaths[0]}"

# Use the go version bazel wants us to use, not system path
GO="$GOROOT/bin/go"

# Find all packages that need generation
files=()
#   first, all packages with the tag, except the tool package itself
white=($(find . -name *.go | \
    grep -v "${tool##*/}" | \
    (xargs grep -l "$match" || true)))
#   now, munge these names a bit:
#     rename /staging/src to /vendor
#     change ./ to k8s.io/kubernetes
#     ignore packages that only have +k8s:foo-gen=false lines
dirs=()
for w in "${white[@]}"; do
  if grep "$match" "$w" | grep -q -v "${match}false"; then
    dirs+=("$(dirname "$w" | sed -e 's|./staging/src/|./vendor/|;s|^./|k8s.io/kubernetes/|')")
  elif [[ -n "${DEBUG:-}" ]]; then
    echo SKIP: $w, has only ${match}false tags
  fi
done

if [[ -n "${DEBUG:-}" ]]; then
  echo "Generating: $(IFS=$'\n ' ; echo "${dirs[*]}" | sort -u)"
else
  echo "Generating $out for ${#dirs} packages..."
fi
# Send the tool the comma-separate list of packages to generate
packages="$(IFS="," ; echo "${dirs[*]}")"
# Run the tool
$dcg \
  -v 1 \
  -i "$packages" \
  -O "${out%%???}" \
  -h "$head" \
  $tool_flags

# DEBUG: Ensure we generated each file
# TODO(fejta): consider deleting this
if [[ -n "${DEBUG:-}" ]]; then
  for p in "${dirs[@]}"; do
    found=false
    for s in "" "k8s.io/kubernetes/vendor/" "staging/src/"; do
      if [[ -f "$srcpath/src/$s$p/$out" ]]; then
        found=true
        if [[ $s == "k8s.io/kubernetes/vendor/" ]]; then
          grep -A 1 import $srcpath/src/$s$p/$out
        fi
        break
      fi
    done
    if [[ $found == false ]]; then
      echo FAILED: $p
      failed=1
    else
      echo FOUND: $p
    fi
  done
  if [[ -n "${failed:-}" ]]; then
    exit 1
  fi
fi

oifs="$IFS"
IFS=$'\n'
# use go list to create lines of pkg import import ...
for line in $($GO list -f '{{.ImportPath}}{{range .Imports}} {{.}}{{end}}' "${dirs[@]}"); do
  pkg="${line%% *}" # first word
  imports="${line#* }" # everything after the first word
  IFS=' '
  deps="$srcpath/src/$pkg/$out.deps"
  echo > "$deps"
  for dep in $imports; do
    if [[ ! "$dep" == k8s.io/kubernetes/* ]]; then
      continue
    fi
    echo //${dep#k8s.io/kubernetes/}:go_default_library >> $deps
  done
  IFS=$'\n'
done
IFS="$oifs"

# copy $1 to where bazel expects to find it
move-out() {
  D="$(OUTS)"
  dst="$O/${D%%/genfiles/*}/genfiles/build/$1/$out"
  options=(
    "k8s.io/kubernetes/$1"
    "${1/vendor\//}"
    "k8s.io/kubernetes/staging/src/${1/vendor\//}"
  )
  found=false
  for o in "${options[@]}"; do
    src="$srcpath/src/$o/$out"
    if [[ ! -f "$src" ]]; then
      continue
    fi
    found=true
    break
  done
  if [[ $found == false ]]; then
    echo "NOT FOUND: $1 in any of the ${options[@]}"
    exit 1
  fi

  if [[ ! -f "$dst" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
    cp -f "$src.deps" "$dst.deps"
    echo ... generated $1/$out
    return 0
  fi
}
