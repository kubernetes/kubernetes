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

DEFAULT_OUTPUT="${KUBE_ROOT}/docs/api-reference"
OUTPUT=${1:-${DEFAULT_OUTPUT}}
# Use REPO_DIR if provided so we can set it to the host-resolvable path
# to the repo root if we are running this script from a container with
# docker mounted in as a volume.
# We pass the host output dir as the source dir to `docker run -v`, but use
# the regular one to compute diff (they will be the same if running this
# test on the host, potentially different if running in a container).
REPO_DIR=${REPO_DIR:-"${KUBE_ROOT}"}
TMP_SUBPATH="_output/generated_html"
OUTPUT_TMP_IN_HOST="${REPO_DIR}/${TMP_SUBPATH}"
OUTPUT_TMP="${KUBE_ROOT}/${TMP_SUBPATH}"

echo "Generating api reference docs at ${OUTPUT_TMP}"

V1_TMP_IN_HOST="${OUTPUT_TMP_IN_HOST}/v1/"
V1_TMP="${OUTPUT_TMP}/v1/"
mkdir -p ${V1_TMP}
V1BETA1_TMP_IN_HOST="${OUTPUT_TMP_IN_HOST}/extensions/v1beta1/"
V1BETA1_TMP="${OUTPUT_TMP}/extensions/v1beta1/"
mkdir -p ${V1BETA1_TMP}
SWAGGER_PATH="${REPO_DIR}/api/swagger-spec/"

echo "Reading swagger spec from: ${SWAGGER_PATH}"

docker run -u $(id -u) --rm -v $V1_TMP_IN_HOST:/output:z -v ${SWAGGER_PATH}:/swagger-source:z gcr.io/google_containers/gen-swagger-docs:v4 \
    v1 \
    https://raw.githubusercontent.com/kubernetes/kubernetes/master/pkg/api/v1/register.go

docker run -u $(id -u) --rm -v $V1BETA1_TMP_IN_HOST:/output:z -v ${SWAGGER_PATH}:/swagger-source:z gcr.io/google_containers/gen-swagger-docs:v4 \
    v1beta1 \
    https://raw.githubusercontent.com/kubernetes/kubernetes/master/pkg/apis/extensions/v1beta1/register.go

# Check if we actually changed anything
pushd "${OUTPUT_TMP}" > /dev/null
touch .generated_html
find . -type f | cut -sd / -f 2- | LC_ALL=C sort > .generated_html
popd > /dev/null

while read file; do
  if [[ -e "${OUTPUT}/${file}" && -e "${OUTPUT_TMP}/${file}" ]]; then
    echo "comparing ${OUTPUT}/${file} with ${OUTPUT_TMP}/${file}"
    # Filter all munges from original content.
    original=$(cat "${OUTPUT}/${file}")
    generated=$(cat "${OUTPUT_TMP}/${file}")

    # Filter out meaningless lines with timestamps
    original=$(echo "${original}" | grep -v "Last updated" || :)
    generated=$(echo "${generated}" | grep -v "Last updated" || :)

    # By now, the contents should be normalized and stripped of any
    # auto-managed content.  
    if diff -Bw >/dev/null <(echo "${original}") <(echo "${generated}"); then
      # actual contents same, overwrite generated with original.
      cp "${OUTPUT}/${file}" "${OUTPUT_TMP}/${file}"
    fi
  fi
done <"${OUTPUT_TMP}/.generated_html"

echo "Moving api reference docs from ${OUTPUT_TMP} to ${OUTPUT}"

cp -af "${OUTPUT_TMP}"/* "${OUTPUT}"
rm -r ${OUTPUT_TMP}

# ex: ts=2 sw=2 et filetype=sh
