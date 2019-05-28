#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

# Script that creates a Kubemark cluster for any given cloud provider.

set -o errexit
set -o nounset
set -o pipefail
set -x
# shellcheck disable=SC2039
KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
# shellcheck source=./skeleton/util.sh
source "${KUBE_ROOT}/test/kubemark/skeleton/util.sh"

# shellcheck source=./cloud-provider-config.sh
source "${KUBE_ROOT}/test/kubemark/cloud-provider-config.sh"

# shellcheck source=./gce/util.sh
source "${KUBE_ROOT}/test/kubemark/${CLOUD_PROVIDER}/util.sh"

# shellcheck source=../../cluster/kubemark/gce/config-default.sh
source "${KUBE_ROOT}/cluster/kubemark/${CLOUD_PROVIDER}/config-default.sh"
set -x
