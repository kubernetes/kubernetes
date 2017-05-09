#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# Usage: ./script/release.sh [TAG]

set -o errexit
set -o nounset
set -o pipefail

base_dir=$(dirname "$0")
base_dir=$(cd "${base_dir}" && pwd)

sleepy_version=${1:-latest}

echo " ---> Cleaning up before building..."
"${base_dir}/clean.sh" 2> /dev/null

echo " ---> Building..."
"${base_dir}/build.sh"

echo " ---> Pushing kubernetes/sleepy:${sleepy_version}..."
"${base_dir}/push.sh" "${sleepy_version}"

echo " ---> Cleaning up..."
"${base_dir}/clean.sh"

echo " ---> Done."
