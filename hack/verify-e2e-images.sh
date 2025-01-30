#!/usr/bin/env bash

# Copyright 2024 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
cd "${KUBE_ROOT}"

source hack/lib/init.sh

# NOTE: Please do NOT add any to this list!!
#
# We are aiming to consolidate on: registry.k8s.io/e2e-test-images/agnhost
# The sources for which are in test/images/agnhost.
# If agnhost is missing functionality for your tests, please reach out to SIG Testing.
kube::util::read-array PERMITTED_IMAGES < <(sed '/^#/d' ./test/images/.permitted-images)

# get current list of images, ignoring tags
echo "Getting e2e image list ..."
make WHAT=test/e2e/e2e.test
e2e_test="$(kube::util::find-binary e2e.test)"
kube::util::read-array IMAGES < <("${e2e_test}" --list-images | sed -E 's/^(.+):[^:]+$/\1/' | LC_ALL=C sort -u)

# diff versus known permitted images
ret=0
>&2 echo "Diffing e2e image list ..."
diff -Naupr <(printf '%s\n' "${IMAGES[@]}") <(printf '%s\n' "${PERMITTED_IMAGES[@]}") || ret=$?
if [[ $ret -eq 0 ]]; then
  >&2 echo "PASS: e2e images used are OK."
else
  >&2 echo "FAIL: e2e images do not match the approved list!"
  >&2 echo ""
  >&2 echo "Please use registry.k8s.io/e2e-test-images/agnhost wherever possible, we are consolidating test images."
  >&2 echo "See: test/images/agnhost/README.md"
  >&2 echo ""
  >&2 echo "You can reach out to https://git.k8s.io/community/sig-testing for help."
  exit 1
fi
