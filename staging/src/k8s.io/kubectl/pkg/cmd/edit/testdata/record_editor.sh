#!/usr/bin/env bash

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

# send the original content to the server
curl -s -k -XPOST "http://localhost:8081/callback/in" --data-binary "@${1}"
# allow the user to edit the file
vi "${1}"
# send the resulting content to the server
curl -s -k -XPOST "http://localhost:8081/callback/out" --data-binary "@${1}"
