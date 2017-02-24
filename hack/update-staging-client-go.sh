#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

V=""
IN_PLACE="0"
COPY_FLAGS="-u"
while getopts ":vidf" opt; do
  case $opt in
    v) # increase verbosity
      V="-v"
      ;;
    i) # do not godep-restore into a temporary directory, but use the existing GOPATH
      IN_PLACE=1
      ;;
    d) # dry-run: do not actually update the client
	  COPY_FLAGS+=" -d"
	  ;;
    f) # fail-on-diff: fail if the client changed
	  COPY_FLAGS+=" -f"
	  ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done
readonly V IN_PLACE

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::util::ensure_godep_version v74

cd ${KUBE_ROOT}

echo "Checking whether godeps are restored"
if ! kube::util::godep_restored 2>&1 | sed 's/^/  /'; then
  echo -e '\nRun 'godep restore' to download dependencies.' 1>&2
  exit 1
fi

echo "Running staging/copy.sh"
eval staging/copy.sh ${COPY_FLAGS} 2>&1 | sed 's/^/  /'
