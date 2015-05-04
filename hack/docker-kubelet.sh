#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

docker run \
  --volume=/:/rootfs:ro \
  --volume=/var/run:/var/run:rw \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --volume=/var/lib/kubelet/:/var/lib/kubelet:rw \
  --net=host \
  --detach=true \
  --privileged=true \
  kubernetes/kubelet:latest \
  /kubelet --v=3 --chaos-chance="0.0" --hostname-override="127.0.0.1" --address="127.0.0.1" --api-servers="127.0.0.1:8080" --port="10250" --resource-container="" --containerized
