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
ret=0

# We need deterministic sort order and want to avoid this extra log output:
#   i18n.go:119] Couldn't find the LC_ALL, LC_MESSAGES or LANG environment variables, defaulting to en_US
export LC_ALL=C

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

# validate "e2e.test --list-images":
# - no unexpected output (whether it's on stderr or stdout)
# - zero exit code (indirectly ensures that tests are set up properly)
#
# For now (https://github.com/google/cadvisor/issues/3707, https://github.com/google/cadvisor/pull/3778)
# we have to ignore additional output caused by cadvisor.
output=$("${e2e_test}" --list-images 2>&1 | grep -v 'factory.go.*Registered Plugin' ) || ret=$?
if [[ $ret -ne 0 ]]; then
  >&2 echo "FAIL: '${e2e_test} --list-images' failed:"
  >&2 echo "${output}"
  exit 1
fi

unexpected_output=$(echo "${output}" | grep -v -E '^([[:alnum:]/.-]+):[^:]+$' || true)
if [[ -n "${unexpected_output}" ]]; then
  >&2 echo "FAIL: '${e2e_test} --list-images' printed unexpected output:"
  >&2 echo "${unexpected_output}"
  exit 1
fi

# extract image names without the version
kube::util::read-array IMAGES < <(echo "${output}" | sed -E 's/^([[:alnum:]/.-]+):[^:]+$/\1/' | sort -u)

# diff versus known permitted images
>&2 echo "Diffing e2e image list ..."
# diff context is irrelevant here because of sorting.
# Instead we want to know about old images (no longer in use, need to be removed)
# and new images (should not get added).
diff <(printf '%s\n' "${PERMITTED_IMAGES[@]}") <(printf '%s\n' "${IMAGES[@]}") | sed -E -e '/^---$/d' -e '/^[[:digit:]]+[acd][[:digit:]]+$/d' -e 's/^</obsolete image:/' -e 's/^>/forbidden image:/' >&2 || ret=$?
if [[ $ret -eq 0 ]]; then
  >&2 echo "PASS: e2e images used are OK."
else
  >&2 echo "FAIL: current e2e images do not match the approved list in test/images/.permitted-images!"
  >&2 echo ""
  >&2 echo "Please use registry.k8s.io/e2e-test-images/agnhost wherever possible, we are consolidating test images."
  >&2 echo "See: test/images/agnhost/README.md"
  >&2 echo ""
  >&2 echo "You can reach out to https://git.k8s.io/community/sig-testing for help."
fi
exit $ret
