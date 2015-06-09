#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

ffjson=$(which ffjson)

if [[ ! -x "$ffjson" ]]; then
  {
    echo "It looks as if you don't have a compiled ffjson binary"
    echo
    echo "Please run 'go get github.com/pquerna/ffjson'"
  } >&2
  exit 1
fi

APIROOT="${KUBE_ROOT}/pkg/api"
TMP_APIROOT="${KUBE_ROOT}/_tmp/api"
_tmp="${KUBE_ROOT}/_tmp"

mkdir -p "${_tmp}"
cp -a "${APIROOT}" "${TMP_APIROOT}"

ffjson ${TMP_APIROOT}/v1beta3/types.go 2> /dev/null
ffjson ${TMP_APIROOT}/v1/types.go 2> /dev/null
ffjson ${TMP_APIROOT}/types.go 2> /dev/null
ffjson ${TMP_APIROOT}/resource/quantity.go 2> /dev/null

echo "diffing ${APIROOT} against freshly generated encoding/decoding"
ret=0
diff -Naupr -I 'source: ' "${APIROOT}" "${TMP_APIROOT}" || ret=$?
cp -a ${TMP_APIROOT} "${KUBE_ROOT}/pkg"
rm -rf "${_tmp}"
if [[ $ret -eq 0 ]]
then
  echo "${APIROOT} up to date."
else
  echo "${APIROOT} is out of date. Please run:"
  echo "ffjson pkg/api/v1/types.go"
  echo "ffjson pkg/api/v1beta3/types.go"
  echo "ffjson pkg/api/types.go"
  echo "ffjson pkg/api/resource/quanity.go"
  exit 1
fi

# ex: ts=2 sw=2 et filetype=sh
