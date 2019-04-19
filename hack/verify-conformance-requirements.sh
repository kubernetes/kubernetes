#!/usr/bin/env bash

# Copyright 2019 The Kubernetes Authors.
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
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::verify_go_version

cd "${KUBE_ROOT}"

errors=()
# Check conformance tests follow the requirements as https://git.k8s.io/community/contributors/devel/sig-architecture/conformance-tests.md#conformance-test-requirements
if ! failedLint=$(go run "${KUBE_ROOT}"/hack/conformance/check_conformance_test_requirements.go "${KUBE_ROOT}"/test/e2e/)
then
  errors+=( "${failedLint}" )
fi

# Check to be sure all the packages that should pass lint are.
if [ ${#errors[@]} -eq 0 ]; then
  echo 'Congratulations!  All e2e test source files have been linted for conformance requirements.'
else
  {
    echo "Errors from lint:"
    for err in "${errors[@]}"; do
      echo "$err"
    done
    echo
    echo 'Please review the above warnings.'
    echo
  } >&2
  exit 1
fi

