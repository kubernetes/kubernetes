#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# Generates updated api-reference docs from the latest swagger spec.
# Usage: ./update-api-reference-docs.sh <absolute output path>

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
kube::golang::setup_env
kube::util::ensure-temp-dir

#KUBE_ROOT should be an absolute path now that we have run kube::golang::setup_env.
REPO_DIR=${REPO_DIR:-"${KUBE_ROOT}"}
DEFAULT_OUTPUT_PATH="${REPO_DIR}/docs/api-reference"
OUTPUT=${1:-${DEFAULT_OUTPUT_PATH}}
OUTPUT_TEMP="${KUBE_TEMP}/generated_html"

echo "Generating api reference docs at ${OUTPUT}"

V1_PATH="${OUTPUT}/v1/"
V1_TEMP_PATH="${OUTPUT_TEMP}/v1/"
mkdir -p ${V1_TEMP_PATH}
V1BETA1_PATH="${OUTPUT}/extensions/v1beta1/"
V1BETA1_TEMP_PATH="${OUTPUT_TEMP}/extensions/v1beta1/"
mkdir -p ${V1BETA1_TEMP_PATH}
SWAGGER_PATH="${REPO_DIR}/api/swagger-spec/"

echo "Reading swagger spec from: ${SWAGGER_PATH}"

mkdir -p $V1_PATH
mkdir -p $V1BETA1_PATH

docker run -u $(id -u) --rm -v $V1_TEMP_PATH:/output:z -v ${SWAGGER_PATH}:/swagger-source:z gcr.io/google_containers/gen-swagger-docs:v4 \
    v1 \
    https://raw.githubusercontent.com/kubernetes/kubernetes/master/pkg/api/v1/register.go

docker run -u $(id -u) --rm -v $V1BETA1_TEMP_PATH:/output:z -v ${SWAGGER_PATH}:/swagger-source:z gcr.io/google_containers/gen-swagger-docs:v4 \
    v1beta1 \
    https://raw.githubusercontent.com/kubernetes/kubernetes/master/pkg/apis/extensions/v1beta1/register.go

# Check if we actually changed anything
pushd "${OUTPUT_TEMP}" > /dev/null
touch .generated_html
find . -type f | cut -sd / -f 2- | LC_ALL=C sort > .generated_html
popd > /dev/null

while read file; do
  if [[ -e "${OUTPUT}/${file}" && -e "${OUTPUT_TEMP}/${file}" ]]; then
    echo "comparing ${OUTPUT}/${file} with ${OUTPUT_TEMP}/${file}"
    # Filter all munges from original content.
    original=$(cat "${OUTPUT}/${file}")
    generated=$(cat "${OUTPUT_TEMP}/${file}")

    # Filter out meaningless lines with timestamps
    original=$(echo "${original}" | grep -v "Last updated" || :)
    generated=$(echo "${generated}" | grep -v "Last updated" || :)

    # By now, the contents should be normalized and stripped of any
    # auto-managed content.  
    if diff -Bw >/dev/null <(echo "${original}") <(echo "${generated}"); then
      # actual contents same, overwrite generated with original.
      cp "${OUTPUT}/${file}" "${OUTPUT_TEMP}/${file}"
    fi
  fi
done <"${OUTPUT_TEMP}/.generated_html"

cp -af "${OUTPUT_TEMP}"/* "${OUTPUT}"

# ex: ts=2 sw=2 et filetype=sh
