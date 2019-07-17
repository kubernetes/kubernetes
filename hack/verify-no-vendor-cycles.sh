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
set -x;

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

export GO111MODULE=auto

staging_repos=()
kube::util::read-array staging_repos < <(kube::util::list_staging_repos)
staging_repos_pattern=$(IFS="|"; echo "${staging_repos[*]}")

cd "${KUBE_ROOT}"

failed=false
while IFS= read -r dir; do
  deps=$(go list -f '{{range .Deps}}{{.}}{{"\n"}}{{end}}' "${dir}" 2> /dev/null || echo "")
  deps_on_main=$(echo "${deps}" | grep -v "k8s.io/kubernetes/vendor/" | grep "k8s.io/kubernetes" || echo "")
  if [ -n "${deps_on_main}" ]; then
    echo "Package ${dir} has a cyclic dependency on the main repository."
    failed=true
  fi
  deps_on_staging=$(echo "${deps}" | grep "k8s.io/kubernetes/vendor/k8s.io" | grep -E "k8s.io\/${staging_repos_pattern}\>" || echo "")
  if [ -n "${deps_on_staging}" ]; then
    echo "Package ${dir} has a cyclic dependency on staging repository packages: ${deps_on_staging}"
    failed=true
  fi
done < <(find ./vendor -type d)

if [[ "${failed}" == "true" ]]; then
  exit 1
fi
