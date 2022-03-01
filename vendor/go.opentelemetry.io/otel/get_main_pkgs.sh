#!/usr/bin/env bash

# Copyright The OpenTelemetry Authors
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

set -euo pipefail

top_dir='.'
if [[ $# -gt 0 ]]; then
    top_dir="${1}"
fi

p=$(pwd)
mod_dirs=()

# Note `mapfile` does not exist in older bash versions:
# https://stackoverflow.com/questions/41475261/need-alternative-to-readarray-mapfile-for-script-on-older-version-of-bash

while IFS= read -r line; do
    mod_dirs+=("$line")
done < <(find "${top_dir}" -type f -name 'go.mod' -exec dirname {} \; | sort)

for mod_dir in "${mod_dirs[@]}"; do
    cd "${mod_dir}"

    while IFS= read -r line; do
        echo ".${line#${p}}"
    done < <(go list --find -f '{{.Name}}|{{.Dir}}' ./... | grep '^main|' | cut -f 2- -d '|')
    cd "${p}"
done
