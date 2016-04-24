#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# The root of the build/dist directory
KUBE_ROOT=$(cd $(dirname "${BASH_SOURCE}")/../.. && pwd -P)

KUBE_OUTPUT_SUBPATH="${KUBE_OUTPUT_SUBPATH:-_output/local}"
KUBE_OUTPUT="${KUBE_ROOT}/${KUBE_OUTPUT_SUBPATH}"
KUBE_OUTPUT_BINPATH="${KUBE_OUTPUT}/bin"

source "${KUBE_ROOT}/hack/lib/util.sh"
source "${KUBE_ROOT}/cluster/lib/util.sh"
source "${KUBE_ROOT}/cluster/lib/logging.sh"

kube::log::install_errexit

source "${KUBE_ROOT}/hack/lib/version.sh"
source "${KUBE_ROOT}/hack/lib/golang.sh"
source "${KUBE_ROOT}/hack/lib/etcd.sh"

KUBE_GIT_UPSTREAM="${KUBE_GIT_UPSTREAM:-upstream}"

KUBE_OUTPUT_HOSTBIN="${KUBE_OUTPUT_BINPATH}/$(kube::util::host_platform)"

# This emulates "readlink -f" which is not available on MacOS X.
# Test:
# T=/tmp/$$.$RANDOM
# mkdir $T
# touch $T/file
# mkdir $T/dir
# ln -s $T/file $T/linkfile
# ln -s $T/dir $T/linkdir
# function testone() {
#   X=$(readlink -f $1 2>&1)
#   Y=$(kube::readlinkdashf $1 2>&1)
#   if [ "$X" != "$Y" ]; then
#     echo readlinkdashf $1: expected "$X", got "$Y"
#   fi
# }
# testone /
# testone /tmp
# testone $T
# testone $T/file
# testone $T/dir
# testone $T/linkfile
# testone $T/linkdir
# testone $T/nonexistant
# testone $T/linkdir/file
# testone $T/linkdir/dir
# testone $T/linkdir/linkfile
# testone $T/linkdir/linkdir
function kube::readlinkdashf {
  # run in a subshell for simpler 'cd'
  (
    if [[ -d "$1" ]]; then # This also catch symlinks to dirs.
      cd "$1"
      pwd -P
    else
      cd $(dirname "$1")
      local f
      f=$(basename "$1")
      if [[ -L "$f" ]]; then
        readlink "$f"
      else
        echo "$(pwd -P)/${f}"
      fi
    fi
  )
}

# This emulates "realpath" which is not available on MacOS X
# Test:
# T=/tmp/$$.$RANDOM
# mkdir $T
# touch $T/file
# mkdir $T/dir
# ln -s $T/file $T/linkfile
# ln -s $T/dir $T/linkdir
# function testone() {
#   X=$(realpath $1 2>&1)
#   Y=$(kube::realpath $1 2>&1)
#   if [ "$X" != "$Y" ]; then
#     echo realpath $1: expected "$X", got "$Y"
#   fi
# }
# testone /
# testone /tmp
# testone $T
# testone $T/file
# testone $T/dir
# testone $T/linkfile
# testone $T/linkdir
# testone $T/nonexistant
# testone $T/linkdir/file
# testone $T/linkdir/dir
# testone $T/linkdir/linkfile
# testone $T/linkdir/linkdir
kube::realpath() {
  if [[ ! -e "$1" ]]; then
    echo "$1: No such file or directory" >&2
    return 1
  fi
  kube::readlinkdashf "$1"
}
