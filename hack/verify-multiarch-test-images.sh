#!/bin/bash

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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

cd "${KUBE_ROOT}"
NON_MULTIARCH_IMAGES=()

for IMAGE_DIR in $(find test/images -iname 'Dockerfile' -printf '%h\n'| LC_ALL=C sort -u)
do
    grep -q "^${IMAGE_DIR}$" hack/.multiarch_images_exceptions.txt && ret=$? || ret=$?
    if [[ "${ret}" -ne 0 ]]; then
        if [[ ! -f "${IMAGE_DIR}/VERSION" ]]; then
            NON_MULTIARCH_IMAGES+=("${IMAGE_DIR}")
        fi
    fi
done

if [ ${#NON_MULTIARCH_IMAGES[@]} -ne 0 ]; then
    echo "[${NON_MULTIARCH_IMAGES[@]}] these images are not written for multiarchitecture."
    echo 'Please port these images to multiarchitectures to avoid this error.'
    echo ''
	echo 'If the above error does not make sense, you can exempt by adding the list to'
	echo 'hack/.multiarch_images_exceptions.txt (if sig/testing is okay with it).'
    exit 1
fi
