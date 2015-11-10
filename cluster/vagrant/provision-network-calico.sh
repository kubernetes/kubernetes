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

DOCKER_BRIDGE=cbr0

function provision-network-calico {
    echo "Installing, enabling prerequisites"
    yum install -y bridge-utils

    # create new docker bridge
    echo "Delete old docker bridge if it exists"
    ip link set dev ${DOCKER_BRIDGE} down || true
    brctl delbr ${DOCKER_BRIDGE} || true
    echo "Create a new docker bridge ${DOCKER_BRIDGE} with IP ${CONTAINER_ADDR} netmask ${CONTAINER_NETMASK}"
    brctl addbr ${DOCKER_BRIDGE}
    ip link set dev ${DOCKER_BRIDGE} up
    ifconfig ${DOCKER_BRIDGE} ${CONTAINER_ADDR} netmask ${CONTAINER_NETMASK} up
    echo "Created docker bridge:"
    ip addr show dev ${DOCKER_BRIDGE}
}
