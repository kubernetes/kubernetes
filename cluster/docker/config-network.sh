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

# This script is used to configure & start flanneld container on bootstrap daemon
# and then configure & restart docker daemon to make network parameters work.

# Users are allowed to re-wirte this script to deploy their own network solution
# instead of flannel. 

set -e

source ~/docker/kube-config/node.env

if [ "$(id -u)" != "0" ]; then
  echo >&2 "config-network.sh must be called as root!"
  exit 1
fi

# Only need to do this on master, so we use MASTER_IP in $1 as a flag
if [[ ! -z $1 ]]; then
  docker -H unix:///var/run/docker-bootstrap.sock run \
    --net=host gcr.io/google_containers/etcd:$ETCD_VERSION \
    etcdctl set /coreos.com/network/config \
    "{ \"Network\": \"$FLANNEL_NET\", \"Backend\": {\"Type\": \"vxlan\"}}"

fi

# Wait for flanneld ready
sleep 3

# We use eth0 for default, may make it configurable in future
flannelCID=$(docker -H unix:///var/run/docker-bootstrap.sock run \
    --restart=always -d --net=host --privileged \
    -v /dev/net:/dev/net quay.io/coreos/flannel:$FLANNEL_VERSION \
    /opt/bin/flanneld --etcd-endpoints=http://${MASTER_IP}:4001 -iface="eth0")

# Copy flannel env out and source it on the host
docker -H unix:///var/run/docker-bootstrap.sock cp ${flannelCID}:/run/flannel/subnet.env .
source subnet.env


DOCKER_CONF=""

# Configure docker net settings, then restart it
# $lsb_dist is detected in kube-deploy/common.sh
# TODO: deal with ubuntu 15.04
# Configure docker net settings, then restart it
case "${lsb_dist}" in
    centos)
        DOCKER_CONF="/etc/sysconfig/docker"
        echo "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | sudo tee -a ${DOCKER_CONF}
        if ! command_exists ifconfig; then
            yum -y -q install net-tools
        fi
        ifconfig docker0 down
        yum -y -q install bridge-utils && brctl delbr docker0 && systemctl restart docker
        ;;
    amzn)
        DOCKER_CONF="/etc/sysconfig/docker"
        echo "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | sudo tee -a ${DOCKER_CONF}
        ifconfig docker0 down
        yum -y -q install bridge-utils && brctl delbr docker0 && service docker restart
        ;;
    ubuntu|debian)
        DOCKER_CONF="/etc/default/docker"
        echo "DOCKER_OPTS=\"\$DOCKER_OPTS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | sudo tee -a ${DOCKER_CONF}
        ifconfig docker0 down
        apt-get install bridge-utils
        brctl delbr docker0
        service docker stop
        while [ `ps aux | grep /usr/bin/docker | grep -v grep | wc -l` -gt 0 ]; do
            echo "Waiting for docker to terminate"
            sleep 1
        done
        service docker start
        ;;
    *)
        echo "Unsupported operations system ${lsb_dist}"
        exit 1
        ;;
esac

# Wait for docker daemon ready
sleep 3

# Verify network 
function verify-network() {
  pgrep -f "/opt/bin/flanneld" >/dev/null 2>&1 || {
    printf "[WARN]: $daemon is not running! \n"        
  }
  printf "\n"
}

# We need to verfiy it here as user is allowed to use their own network solution instead of flannel,
# so we can not verify network afterwards
verify-network

