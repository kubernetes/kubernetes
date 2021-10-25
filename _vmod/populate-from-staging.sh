#!/usr/bin/env bash

# Copyright 20201 The Kubernetes Authors.
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

#FIXME: call this during build
#FIXME: call cleanup during make clean
#FIXME: exclude more files: code-of-conduct.md CONTRIBUTING.md LICENSE
KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
cd "${KUBE_ROOT}"

for LEAF in $(ls staging/src/k8s.io); do
    SDIR="staging/src/k8s.io/${LEAF}"
    VDIR="_vmod/k8s.io/${LEAF}"
    mkdir -p "${VDIR}"

    # clean up
    find "${VDIR}" -maxdepth 1 -type l | xargs rm -f

    # relink everything
    for X in $(ls "${SDIR}"); do
        ln -sf "../../../${SDIR}/${X}" "${VDIR}/${X}"
    done
done
