#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# This script is intended to start the kubelet and then loop until
# it detects a failure.  It then exits, and supervisord restarts it
# which in turn restarts the kubelet.

{% set kubelet_port = "10250" -%}
{% if pillar['kubelet_port'] is defined -%}
	{% set kubelet_port = pillar['kubelet_port'] -%}
{% endif -%}

/etc/init.d/kubelet stop
/etc/init.d/kubelet start

echo "waiting a minute for startup"
sleep 60

max_seconds=10

while true; do
  if ! curl --insecure -m ${max_seconds} -f -s https://127.0.0.1:{{kubelet_port}}/healthz > /dev/null; then
    echo "kubelet failed!"
    curl --insecure https://127.0.0.1:{{kubelet_port}}/healthz
    exit 2
  fi
  sleep 10
done

