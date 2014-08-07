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

set -e

cd $(dirname ${BASH_SOURCE})/../../

# start the cluster with 2 minions
export KUBERNETES_NUM_MINIONS=2
export KUBERNETES_PROVIDER=vagrant
cluster/kube-up.sh

echo "Pull an image that runs a web server"
vagrant ssh minion-1 -- sudo docker pull dockerfile/nginx
vagrant ssh minion-2 -- sudo docker pull dockerfile/nginx

echo "Run the servers"
vagrant ssh minion-1 -- sudo docker run -d dockerfile/nginx
vagrant ssh minion-2 -- sudo docker run -d dockerfile/nginx

echo "Run ping from minion-1 to docker bridges and to the containers on both minions"
vagrant ssh minion-1 -- 'ping -c 10 10.244.1.1 && ping -c 10 10.244.2.1 && ping -c 10 10.244.1.3 && ping -c 10 10.244.2.3'
echo "Same pinch from minion-2"
vagrant ssh minion-2 -- 'ping -c 10 10.244.1.1 && ping -c 10 10.244.2.1 && ping -c 10 10.244.1.3 && ping -c 10 10.244.2.3'

echo "tcp check, curl to both the running webservers from both machines"
vagrant ssh minion-1 -- 'curl 10.244.1.3:80  && curl 10.244.2.3:80'
vagrant ssh minion-2 -- 'curl 10.244.1.3:80 && curl 10.244.2.3:80'

echo "All good, destroy the cluster"
vagrant destroy -f
