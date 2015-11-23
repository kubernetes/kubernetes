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

# Useful for testing images and changes, turns up a fresh single node cluster

set -o errexit
set -o nounset
set -o pipefail

docker run --net=host -d gcr.io/google_containers/etcd:2.2.1 \
  /usr/local/bin/etcd \
    --addr=127.0.0.1:4001 \
    --bind-addr=0.0.0.0:4001 \
    --data-dir=/var/etcd/data

docker run --pid=host \
  --volume=/:/rootfs:ro \
  --volume=/sys:/sys:ro \
  --volume=/dev:/dev \
  --volume=/var/lib/docker/:/var/lib/docker:rw \
  --volume=/var/lib/kubelet/:/var/lib/kubelet:rw \
  --volume=/var/run:/var/run:rw \
  --net=host \
  --pid=host \
  --privileged=true \
  -d gcr.io/google_containers/hyperkube:v${K8S_VERSION} \
  /hyperkube kubelet \
    --containerized \
    --hostname-override="127.0.0.1" \
    --address="0.0.0.0" \
    --api-servers=http://localhost:8080 \
    --config=/etc/kubernetes/manifests --v=10

docker run -d --net=host --privileged \
  gcr.io/google_containers/hyperkube:v${K8S_VERSION} \
  /hyperkube proxy \
    --master=http://127.0.0.1:8080 --v=2
