#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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


if command -v git &>/dev/null; then
    git diff --no-index "${1}" "${2}" |grep containerPort -A 1 > "./git-diff.result"
elif command -v diff &>/dev/null; then
    diff "${1}" "${2}" |grep containerPort > "./diff.result"
else
    echo "git and diff are unavailable" >&2
    exit 1
fi