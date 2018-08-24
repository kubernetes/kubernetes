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

CURRENT_DIR=$(dirname $0)
OUTPUT_FOLDER="${CURRENT_DIR}/output"

pushd "${CURRENT_DIR}" >/dev/null
  mkdir -p "${OUTPUT_FOLDER}"

  for generator_dir in cmd/*; do
    executable_name="${generator_dir##*/}"
    echo "Building ${executable_name}..."
    go build -o "${OUTPUT_FOLDER}/${executable_name}" "./${generator_dir}"
  done
popd >/dev/null

echo "Building complete. Artifacts are in ${OUTPUT_FOLDER} folder."
