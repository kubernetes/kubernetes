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

# GoFmt apparently is changing @ head...

GO_VERSION=($(go version))
echo "Detected go version: $(go version)"

if [[ ${GO_VERSION[2]} != "go1.2" && ${GO_VERSION[2]} != "go1.3" ]]; then
  echo "Unknown go version, skipping gofmt."
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/../" && pwd -P)"

files="$(find ${REPO_ROOT} -type f | grep "[.]go$" | grep -v "third_party/\|release/\|_?output/\|target/\|Godeps/")"
bad=$(gofmt -s -l ${files})
if [[ -n "${bad}" ]]; then
  echo "$bad"
  exit 1
fi
