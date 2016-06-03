#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

KUBE_ROOT=$(readlink -m $(dirname "${BASH_SOURCE}")/../../)

. ${KUBE_ROOT}/federation/cluster/common.sh

tagfile="${KUBE_ROOT}/federation/manifests/federated-image.tag"
if [[ ! -f "$tagfile" ]]; then
    echo "FATAL: tagfile ${tagfile} does not exist. Make sure that you have run build/push-federation-images.sh"
    exit 1
fi
export FEDERATION_IMAGE_TAG="$(cat "${KUBE_ROOT}/federation/manifests/federated-image.tag")"

create-federation-api-objects
