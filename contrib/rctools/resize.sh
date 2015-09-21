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

# This command resizes a replication controller using kubectl.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"

if [[ $# != 2 ]] ; then
  echo "usage: $0 <replication controller name> <size>" >&2
  exit 1
fi

rc="$1"
size="$2"

"${KUBECTL}" get -o json rc "$rc" | sed 's/"replicas": [0-9][0-9]*/"replicas": '"$size"'/' | "${KUBECTL}" update -f - rc "$rc"
