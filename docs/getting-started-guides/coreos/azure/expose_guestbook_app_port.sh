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

set -e

[ ! -z $1 ] || (echo Usage: $0 ssh_conf; exit 1)

fe_port=$(ssh -F $1 kube-00 \
  "/opt/bin/kubectl get -o template --template='{{(index .spec.ports 0).nodePort}}' services frontend -L name=frontend" \
)

echo "Guestbook app is on port $fe_port, will map it to port 80 on kube-00"

./node_modules/.bin/azure vm endpoint create kube-00 80 $fe_port

./node_modules/.bin/azure vm endpoint show kube-00 tcp-80-${fe_port}
