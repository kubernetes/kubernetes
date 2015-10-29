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
# Author @wizard_cxy @reouser

set -e

# Make sure docker daemon is running
if ( ! ps -ef | grep "/usr/bin/docker" | grep -v 'grep' &> /dev/null ); then
    echo "Docker is not running on this machine!"
    exit 1
fi

# Make sure k8s version env is properly set
if [ -z ${K8S_VERSION} ]; then
    K8S_VERSION="1.0.3"
    echo "K8S_VERSION is not set, using default: ${K8S_VERSION}"
else
    echo "k8s version is set to: ${K8S_VERSION}"
fi


# Run as root
if [ "$(id -u)" != "0" ]; then
    echo >&2 "Please run as root"
    exit 1
fi

# Check if a command is valid
command_exists() {
    command -v "$@" > /dev/null 2>&1
}

lsb_dist=""

# Detect the OS distro, we support ubuntu, debian, mint, centos, fedora dist
detect_lsb() {
    case "$(uname -m)" in
    *64)
        ;;
    *)
        echo "Error: We currently only support 64-bit platforms."       
        exit 1
        ;;
    esac

    if command_exists lsb_release; then
        lsb_dist="$(lsb_release -si)"
    fi
    if [ -z ${lsb_dist} ] && [ -r /etc/lsb-release ]; then
        lsb_dist="$(. /etc/lsb-release && echo "$DISTRIB_ID")"
    fi
    if [ -z ${lsb_dist} ] && [ -r /etc/debian_version ]; then
        lsb_dist='debian'
    fi
    if [ -z ${lsb_dist} ] && [ -r /etc/fedora-release ]; then
        lsb_dist='fedora'
    fi
    if [ -z ${lsb_dist} ] && [ -r /etc/os-release ]; then
        lsb_dist="$(. /etc/os-release && echo "$ID")"
    fi

    lsb_dist="$(echo ${lsb_dist} | tr '[:upper:]' '[:lower:]')"
}


# Start the bootstrap daemon
bootstrap_daemon() {
    sudo -b docker -d -H unix:///var/run/docker-bootstrap.sock -p /var/run/docker-bootstrap.pid --iptables=false --ip-masq=false --bridge=none --graph=/var/lib/docker-bootstrap 2> /var/log/docker-bootstrap.log 1> /dev/null
    
    sleep 5
}

# Start k8s components in containers
DOCKER_CONF=""

start_k8s(){
    # Start etcd 
    docker -H unix:///var/run/docker-bootstrap.sock run --restart=always --net=host -d gcr.io/google_containers/etcd:2.0.12 /usr/local/bin/etcd --addr=127.0.0.1:4001 --bind-addr=0.0.0.0:4001 --data-dir=/var/etcd/data

    sleep 5
    # Set flannel net config
    docker -H unix:///var/run/docker-bootstrap.sock run --net=host gcr.io/google_containers/etcd:2.0.12 etcdctl set /coreos.com/network/config '{ "Network": "10.1.0.0/16", "Backend": {"Type": "vxlan"}}'

    # iface may change to a private network interface, eth0 is for default
    flannelCID=$(docker -H unix:///var/run/docker-bootstrap.sock run --restart=always -d --net=host --privileged -v /dev/net:/dev/net quay.io/coreos/flannel:0.5.0 /opt/bin/flanneld -iface="eth0")

    sleep 8

    # Copy flannel env out and source it on the host
    docker -H unix:///var/run/docker-bootstrap.sock cp ${flannelCID}:/run/flannel/subnet.env .
    source subnet.env

    # Configure docker net settings, then restart it
    case "$lsb_dist" in
        fedora|centos|amzn)
            DOCKER_CONF="/etc/sysconfig/docker"
        ;;
        ubuntu|debian|linuxmint)
            DOCKER_CONF="/etc/default/docker"
        ;;
    esac

    # Append the docker opts
    echo "DOCKER_OPTS=\"\$DOCKER_OPTS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | sudo tee -a ${DOCKER_CONF}


    # sleep a little bit
    ifconfig docker0 down

    case "$lsb_dist" in
        fedora|centos|amzn)
            yum install bridge-utils && brctl delbr docker0 && systemctl restart docker
        ;;
        ubuntu|debian|linuxmint)
            apt-get install bridge-utils && brctl delbr docker0 && service docker restart
        ;;
    esac

    # sleep a little bit
    sleep 5

    # Start kubelet & proxy, then start master components as pods
    docker run \
        --net=host \
        --pid=host \
        --privileged \
        --restart=always \
        -d \
        -v /sys:/sys:ro \
        -v /var/run:/var/run:rw \
        -v /:/rootfs:ro \
        -v /dev:/dev \
        -v /var/lib/docker/:/var/lib/docker:ro \
        -v /var/lib/kubelet/:/var/lib/kubelet:rw \
        gcr.io/google_containers/hyperkube:v${K8S_VERSION} \
        /hyperkube kubelet \
        --api-servers=http://localhost:8080 \
        --v=2 --address=0.0.0.0 --enable-server \
        --hostname-override=127.0.0.1 \
        --config=/etc/kubernetes/manifests-multi \
        --cluster-dns=10.0.0.10 \
        --cluster-domain=cluster.local
    
    docker run \
        -d \
        --net=host \
        --privileged \
        gcr.io/google_containers/hyperkube:v${K8S_VERSION} \
        /hyperkube proxy --master=http://127.0.0.1:8080 --v=2   
}

echo "Detecting your OS distro ..."
detect_lsb

echo "Starting bootstrap docker ..."
bootstrap_daemon

echo "Starting k8s ..."
start_k8s

echo "Master done!"
