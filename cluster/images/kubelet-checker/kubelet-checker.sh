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

# This script is intended to watch kubelet health and restart it when
# it detects a failure.
#
# Usage:
#  kubelet-checker.sh <kubelet_container_id> [<kubelet_url>]

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
	echo "Usage:"
	echo "  kubelet-checker.sh <kubelet_container_id> [<kubelet_url>]"
	exit 1
fi

max_seconds=10
container_id="${1}"
kubelet_url="${2:-https://127.0.0.1:10250}"

echo "Watching kubelet on ${kubelet_url} (container_id=${container_id})"
echo ""
echo "Waiting a minute for kubelet startup"
sleep 60
while true; do
  if ! curl --insecure -m ${max_seconds} -f -s ${kubelet_url}/healthz > /dev/null; then
    echo "Kubelet failed!"
    sudo docker restart $1
    echo "Waiting a minute for kubelet startup"
	sleep 60
  else
	sleep 10
  fi
done

