#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

if [[ $# -ne 1 ]]; then
    echo 'use "bazel run //hack:update-mirror"'
    echo "(usage: $0 <file with list of URLs to mirror>)"
    exit 1
fi

BUCKET="gs://k8s-bazel-cache"

gsutil acl get "${BUCKET}" > /dev/null

tmpfile=$(mktemp bazel_workspace_mirror.XXXXXX)
trap 'rm ${tmpfile}' EXIT
while read -r url; do
  echo "${url}"
  if gsutil ls "${BUCKET}/${url}" &> /dev/null; then
    echo present
  else
    echo missing
    if curl -fLag "${url}" > "${tmpfile}"; then
        gsutil cp -a public-read "${tmpfile}" "${BUCKET}/${url}"
    fi
  fi
done < "$1"
