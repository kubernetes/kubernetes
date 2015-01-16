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

NEW_SIZE=${1:-4}

export KUBE_REPO_ROOT=${KUBE_REPO_ROOT-$(dirname $0)/../..}
export KUBECTL=${KUBECTL-$KUBE_REPO_ROOT/cluster/kubectl.sh}

set -x

$KUBECTL resize rc update-demo-nautilus --replicas=$NEW_SIZE
