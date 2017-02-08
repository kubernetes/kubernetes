#!/usr/bin/env bash
# Copyright 2017 The Kubernetes Authors.
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

export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
cd "${KUBE_ROOT}"
# List go packages with the k8s:openapi-gen tag.
files=$(grep --color=never -l '+k8s:openapi-gen=' \
    $(find . -name "*.go") \
    | xargs -n1 dirname \
    | LC_ALL=C sort -u \
    | grep -v "/staging/")

# Verify that they are included in the openapi genrule.
rc=0
def="pkg/generated/openapi/def.bzl"
for f in ${files}; do
  nodot=${f#./}
  novendor=${nodot#vendor/}
  if ! grep "${novendor}" "${def}" &> /dev/null; then
    echo "Package not listed in ${def}: ${novendor}"
    rc=1
  fi
done
exit ${rc}
