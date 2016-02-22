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

# A scripts to install k8s worker node.

set -e

source ~/docker/kube-deploy/common.sh

# Start k8s components in containers
function start_k8s() {

    start-network

    $DESTROY_SH clear_old_components

    # start kubelet
    docker run --net=host \
        --privileged=true \
        --restart=always \
        -d \
        -v /:/rootfs:ro \
        -v /sys:/sys:ro \
        -v /dev:/dev \
        -v /var/lib/docker/:/var/lib/docker:ro \
        -v /var/lib/kubelet/:/var/lib/kubelet:rw \
        -v /var/run:/var/run:rw \
        --name=kube_in_docker_kubelet_$RANDOM \
        gcr.io/google_containers/hyperkube:v$K8S_VERSION \
        /hyperkube kubelet \
        --containerized \
        --api-servers=http://$MASTER_IP:8080 \
        --cluster-dns=$DNS_SERVER_IP \
        --cluster-domain=$DNS_DOMAIN \
        $KUBELET_OPTS

    docker run -d --net=host --restart=always --privileged \
        --name kube_in_docker_proxy_$RANDOM \
        gcr.io/google_containers/hyperkube:v$K8S_VERSION \
        /hyperkube proxy --master=http://$MASTER_IP:8080 --v=2
}
echo "... Detecting your OS distro"
detect_lsb
echo "... Starting bootstrap daemon"
bootstrap_daemon
echo "... Starting k8s"
start_k8s
echo "Worker done!"

