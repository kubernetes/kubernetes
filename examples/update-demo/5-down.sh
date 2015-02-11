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

export KUBE_ROOT=$(dirname $0)/../..
export KUBECTL=${KUBE_ROOT}/cluster/kubectl.sh

set -x

rc="update-demo-kitten"
echo "Stopping replicationController ${rc}"
$KUBECTL stop rc ${rc} || true

rc="update-demo-nautilus"
echo "Stopping replicationController ${rc}"
$KUBECTL stop rc ${rc} || true
