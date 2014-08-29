#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

REPO_ROOT="$(cd "$(dirname "$0")/../" && pwd -P)"

result=0

gofiles="$(find ${REPO_ROOT} -type f | grep "[.]go$" | grep -v "Godeps/\|third_party/\|release/\|_?output/|target/")"
for file in ${gofiles}; do
  if [[ "$(${REPO_ROOT}/hooks/boilerplate.sh "${file}")" -eq "0" ]]; then
    echo "Boilerplate header is wrong for: ${file}"
    result=1
  fi
done

dirs=("cluster" "hack" "hooks")

for dir in ${dirs[@]}; do
  for file in $(grep -r -l "" "${REPO_ROOT}/${dir}/" | grep "[.]sh"); do
    if [[ "$(${REPO_ROOT}/hooks/boilerplate.sh "${file}")" -eq "0" ]]; then
      echo "Boilerplate header is wrong for: ${file}"
      result=1
    fi
  done
done


exit ${result}
