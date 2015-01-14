#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env
"${KUBE_ROOT}/hack/build-go.sh" cmd/gendocs

# Get the absolute path of the directory component of a file, i.e. the
# absolute path of the dirname of $1.
get_absolute_dirname() {
  echo "$(cd "$(dirname "$1")" && pwd)"
}

# Detect the OS name/arch so that we can find our binary
case "$(uname -s)" in
  Darwin)
    host_os=darwin
    ;;
  Linux)
    host_os=linux
    ;;
  *)
    echo "Unsupported host OS.  Must be Linux or Mac OS X." >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64*)
    host_arch=amd64
    ;;
  i?86_64*)
    host_arch=amd64
    ;;
  amd64*)
    host_arch=amd64
    ;;
  arm*)
    host_arch=arm
    ;;
  i?86*)
    host_arch=x86
    ;;
  *)
    echo "Unsupported host arch. Must be x86_64, 386 or arm." >&2
    exit 1
    ;;
esac

# Find binary
locations=(
  "${KUBE_ROOT}/_output/dockerized/bin/${host_os}/${host_arch}/gendocs"
  "${KUBE_ROOT}/_output/local/bin/${host_os}/${host_arch}/gendocs"
  "${KUBE_ROOT}/platforms/${host_os}/${host_arch}/gendocs"
)
gendocs=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )

if [[ ! -x "$gendocs" ]]; then
  {
    echo "It looks as if you don't have a compiled gendocs binary"
    echo
    echo "If you are running from a clone of the git repo, please run"
    echo "'./hack/build-go.sh cmd/gendocs'."
  } >&2
  exit 1
fi


KUBECTL_DOC="docs/kubectl.md"

echo "diffing ${KUBECTL_DOC} against generated output from ${gendocs}"
"${gendocs}" | diff "${KUBE_ROOT}/${KUBECTL_DOC}" - && echo "${KUBECTL_DOC} up to date." || {
  echo "${KUBECTL_DOC} is out of date. Please run ${gendocs} > ${KUBECTL_DOC}"
  exit 1
}
