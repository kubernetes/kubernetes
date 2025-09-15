#!/usr/bin/env bash

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

# This script checks whether the expected metadata files (such as OWNERS and
# LICENSE) exist under the `staging/src/k8s.io/*` directories.
# Usage: `hack/verify-staging-meta-files.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

expected_filenames=(
  .github/PULL_REQUEST_TEMPLATE.md
  code-of-conduct.md
  LICENSE
  OWNERS
  README.md
  SECURITY_CONTACTS
)

exceptions=(
  client-go/README.md # client-go provides its own README
)

RESULT=0
for full_repo_path in "${KUBE_ROOT}"/staging/src/k8s.io/*; do
  repo=$(basename "${full_repo_path}")
  for filename in "${expected_filenames[@]}"; do
    if echo " ${exceptions[*]} " | grep -F " ${repo}/${filename} " >/dev/null; then
      continue
    elif [ ! -f "${KUBE_ROOT}/staging/src/k8s.io/${repo}/${filename}" ]; then
      echo "staging/src/k8s.io/${repo}/${filename} does not exist and must be created"
      RESULT=1
    fi
  done
done

exit $RESULT
