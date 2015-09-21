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

# This command resizes a replication controller to 0.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
RESIZE="${KUBE_ROOT}/contrib/rctools/resize.sh"

if [[ $# != 1 ]] ; then
  echo "usage: $0 <replication controller name>" >&2
  exit 1
fi

rc="$1"

"${RESIZE}" "$rc" 0

# kubectl describe output includes a line like:
# Replicas:       2 current / 2 desired

# Wait until it shows 0 pods
while true; do
  pods=$(${KUBECTL} describe rc "$rc" | awk '/^Replicas:/{print $2}')
  if [[ "$pods" -eq 0 ]] ; then
    exit 0
  else
    echo "$pods remaining..."
  fi
  sleep 1
done

exit 1
