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

# Marks the current stable version

set -o errexit
set -o nounset
set -o pipefail

if [[ -z "$1" ]]; then
  echo "Usage: $0 <version>"
  exit 1
fi

if ! gsutil ls gs://kubernetes-release/release/${1}/kubernetes.tar.gz; then
  echo "Release files don't exist, aborting."
  exit 2
fi

STABLE_FILE_LOCATION="kubernetes-release/release/stable.txt"

version_file=$(mktemp -t stable.XXXXXX)

echo $1 >> ${version_file}
echo "Uploading stable version $1 to google storage"
gsutil cp ${version_file} "gs://${STABLE_FILE_LOCATION}"
echo "Making it world readable"
gsutil acl ch -R -g all:R "gs://${STABLE_FILE_LOCATION}"

rm ${version_file}

value=$(curl -s https://storage.googleapis.com/${STABLE_FILE_LOCATION})
echo "Validating version file"
if [[ "${value}" != "${1}" ]]; then
  echo "Error validating upload, :${value}: vs expected :${1}:"
  exit 1
fi
