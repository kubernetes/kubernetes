#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

function usage {
  local script="$(basename $0)"

  echo >&2 "Usage: ${script} [-r <branch|tag> | -d <dir>]

This script should be run at the root of a module.

-r <branch|tag>
  Compare the exported API of the local working copy with the 
  exported API of the local repo at the specified branch or tag.

-d <dir>
  Compare the exported API of the local working copy with the 
  exported API of the specified directory, which should point
  to the root of a different version of the same module.

Examples:
  ${script} -r master
  ${script} -r v1.10.0
  ${script} -r release-1.10
  ${script} -d /path/to/historical/version
"
  exit 1
}

ref=""
dir=""
while getopts r:d: o
do case "$o" in
  r)    ref="$OPTARG";;
  d)    dir="$OPTARG";;
  [?])  usage;;
  esac
done

# If REF and DIR are empty, print usage and error
if [[ -z "${ref}" && -z "${dir}" ]]; then
  usage;
fi
# If REF and DIR are both set, print usage and error
if [[ -n "${ref}" && -n "${dir}" ]]; then
  usage;
fi

if ! which apidiff > /dev/null; then
  echo "Installing golang.org/x/exp/cmd/apidiff..."
  pushd "${TMPDIR:-/tmp}" > /dev/null
    go get golang.org/x/exp/cmd/apidiff
  popd > /dev/null
fi

output=$(mktemp -d -t "apidiff.output.XXXX")
cleanup_output () { rm -fr "${output}"; }
trap cleanup_output EXIT

# If ref is set, clone . to temp dir at $ref, and set $dir to the temp dir
clone=""
base="${dir}"
if [[ -n "${ref}" ]]; then
  base="${ref}"
  clone=$(mktemp -d -t "apidiff.clone.XXXX")
  cleanup_clone_and_output () { rm -fr "${clone}"; cleanup_output; }
  trap cleanup_clone_and_output EXIT
  git clone . -q --no-tags -b "${ref}" "${clone}"
  dir="${clone}"
fi

pushd "${dir}" >/dev/null
  echo "Inspecting API of ${base}..."
  go list ./... > packages.txt
  for pkg in $(cat packages.txt); do
    mkdir -p "${output}/${pkg}"
    apidiff -w "${output}/${pkg}/apidiff.output" "${pkg}"
  done
popd >/dev/null

retval=0

echo "Comparing with ${base}..."
for pkg in $(go list ./...); do
  # New packages are ok
  if [ ! -f "${output}/${pkg}/apidiff.output" ]; then
    continue
  fi

  # Check for incompatible changes to previous packages
  incompatible=$(apidiff -incompatible "${output}/${pkg}/apidiff.output" "${pkg}")
  if [[ -n "${incompatible}" ]]; then
    echo >&2 "FAIL: ${pkg} contains incompatible changes:
${incompatible}
"
    retval=1
  fi
done

# Check for removed packages
removed=$(comm -23 "${dir}/packages.txt" <(go list ./...))
if [[ -n "${removed}" ]]; then
  echo >&2 "FAIL: removed packages:
${removed}
"
  retval=1
fi

exit $retval
