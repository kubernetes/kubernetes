#!/bin/bash

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

# Generates updated api-reference docs from the latest swagger spec.
# Usage: ./update-api-reference-docs.sh <absolute output path>

set -o errexit
set -o nounset
set -o pipefail

echo "Note: This assumes that swagger spec has been updated. Please run hack/update-swagger-spec.sh to ensure that."

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

DEFAULT_GROUP_VERSIONS="v1 extensions/v1beta1 batch/v1 autoscaling/v1 certificates/v1alpha1"
VERSIONS=${VERSIONS:-$DEFAULT_GROUP_VERSIONS}
for ver in $VERSIONS; do
  mkdir -p "${OUTPUT_TMP}/${ver}"
done

SWAGGER_PATH="${REPO_DIR}/api/swagger-spec/"

echo "Reading swagger spec from: ${SWAGGER_PATH}"

user_flags="-u $(id -u)"
if [[ $(uname) == "Darwin" ]]; then
  # mapping in a uid from OS X doesn't make any sense
  user_flags=""
fi

for ver in $VERSIONS; do
  TMP_IN_HOST="${OUTPUT_TMP_IN_HOST}/${ver}"
  if [[ ${ver} == "v1" ]]; then
    REGISTER_FILE="${REPO_DIR}/pkg/api/${ver}/register.go"
  else
    REGISTER_FILE="${REPO_DIR}/pkg/apis/${ver}/register.go"
  fi
  SWAGGER_JSON_NAME="$(kube::util::gv-to-swagger-name "${ver}")"

  docker run ${user_flags} \
    --rm -v "${TMP_IN_HOST}":/output:z \
    -v "${SWAGGER_PATH}":/swagger-source:z \
    -v "${REGISTER_FILE}":/register.go:z \
    --net=host -e "https_proxy=${KUBERNETES_HTTPS_PROXY:-}" \
    gcr.io/google_containers/gen-swagger-docs:v8 \
    "${SWAGGER_JSON_NAME}"
done

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
    if diff -B >/dev/null <(echo "${original}") <(echo "${generated}"); then
      # actual contents same, overwrite generated with original.
      cp "${OUTPUT}/${file}" "${OUTPUT_TMP}/${file}"
    fi
  fi
done <"${OUTPUT_TMP}/.generated_html"

echo "Moving api reference docs from ${OUTPUT_TMP} to ${OUTPUT}"

cp -af "${OUTPUT_TMP}"/* "${OUTPUT}"
rm -r "${OUTPUT_TMP}"

# ex: ts=2 sw=2 et filetype=sh
