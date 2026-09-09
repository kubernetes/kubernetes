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

# This script checks the documentation (description) from the OpenAPI specification for URLs, and
# verifies those URLs are valid.
# Usage: `hack/verify-openapi-docs-links.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

SPECROOT="${KUBE_ROOT}/api/openapi-spec"
SPECV3PATH="${SPECROOT}/v3"

_tmpdir="$(kube::realpath "$(mktemp -d -t "$(basename "$0").XXXXXX")")"
mkdir -p "${_tmpdir}"
trap 'rm -rf ${_tmpdir}' EXIT SIGINT
trap "echo Aborted; exit;" SIGINT SIGTERM

TMP_URLS="${_tmpdir}/docs_urls.txt"
touch "${TMP_URLS}"


for full_repo_path in "${SPECV3PATH}"/*.json; do
  grep -oE '"description": ".*",?$' "$full_repo_path" | grep -oE 'https?:\/\/[-a-zA-Z0-9._+]{1,256}\.[a-zA-Z0-9]{1,6}\b([-a-zA-Z0-9:%_+.~&/=]*[a-zA-Z0-9])' >> "${TMP_URLS}" || true
done
sort -u "${TMP_URLS}" -o "${TMP_URLS}"

RESULT=0
while read -r URL; do
  if ! curl --head --location --fail --silent "$URL" > /dev/null; then
    echo "$URL not found"
    RESULT=1
  fi
done < "${TMP_URLS}"

exit $RESULT

# ex: ts=2 sw=2 et filetype=sh
