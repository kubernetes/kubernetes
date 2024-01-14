#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
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

# This script checks wheather any blacklisted words have been added in the diff
# between the commits on the current branch and master

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

cd "${KUBE_ROOT}"

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}

# Issue that explains in detail about disabling 
# "NoOptDefVal" for future PRs: https://github.com/kubernetes/kubectl/issues/1442
blocklisted=("NoOptDefVal")
kubectl_project="staging/src/k8s.io/kubectl/"

kube::util::contains_blocklisted_words $blocklisted $kubectl_project $branch
