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

# A script to setup the k8s master in docker containers.
# Authors @wizard_cxy @resouer

set -e

# Make sure docker daemon is running
if ( ! ps -ef | grep "/usr/bin/docker" | grep -v 'grep' &> /dev/null ); then
    echo "Docker is not running on this machine!"
    exit 1
fi

# Make sure k8s version env is properly set
K8S_VERSION=${K8S_VERSION:-"1.2.0-alpha.7"}
ETCD_VERSION=${ETCD_VERSION:-"2.2.1"}
FLANNEL_VERSION=${FLANNEL_VERSION:-"0.5.5"}
FLANNEL_IPMASQ=${FLANNEL_IPMASQ:-"true"}
FLANNEL_IFACE=${FLANNEL_IFACE:-"eth0"}
ARCH=${ARCH:-"amd64"}

# Run as root
if [ "$(id -u)" != "0" ]; then
    echo >&2 "Please run as root"
    exit 1
fi

# Make sure master ip is properly set
if [ -z ${MASTER_IP} ]; then
    MASTER_IP=$(hostname -I | awk '{print $1}')
fi

echo "K8S_VERSION is set to: ${K8S_VERSION}"
echo "ETCD_VERSION is set to: ${ETCD_VERSION}"
echo "FLANNEL_VERSION is set to: ${FLANNEL_VERSION}"
echo "FLANNEL_IFACE is set to: ${FLANNEL_IFACE}"
echo "FLANNEL_IPMASQ is set to: ${FLANNEL_IPMASQ}"
echo "MASTER_IP is set to: ${MASTER_IP}"
echo "ARCH is set to: ${ARCH}"

# Check if a command is valid
command_exists() {
    command -v "$@" > /dev/null 2>&1
}

lsb_dist=""

# Detect the OS distro, we support ubuntu, debian, mint, centos, fedora dist
detect_lsb() {
    # TODO: remove this when ARM support is fully merged
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

    case "${lsb_dist}" in
        amzn|centos|debian|ubuntu)
            ;;
        *)
            echo "Error: We currently only support ubuntu|debian|amzn|centos."
            exit 1
            ;;
    esac
}


# Start the bootstrap daemon
# TODO: do not start docker-bootstrap if it's already running
bootstrap_daemon() {
    # Detecting docker version so we could run proper docker_daemon command
    [[ $(eval "docker --version") =~ ([0-9][.][0-9][.][0-9]*) ]] && version="${BASH_REMATCH[1]}"
    local got=$(echo -e "${version}\n1.8.0" | sed '/^$/d' | sort -nr | head -1)
    if [[ "${got}" = "${version}" ]]; then
        docker_daemon="docker -d"
    else
        docker_daemon="docker daemon"
    fi
    ${docker_daemon} \
        -H unix:///var/run/docker-bootstrap.sock \
        -p /var/run/docker-bootstrap.pid \
        --iptables=false \
        --ip-masq=false \
        --bridge=none \
        --graph=/var/lib/docker-bootstrap \
            2> /var/log/docker-bootstrap.log \
            1> /dev/null &

    sleep 5
}

# Start k8s components in containers
DOCKER_CONF=""

start_k8s(){
    # Start etcd
    docker -H unix:///var/run/docker-bootstrap.sock run \
        --restart=on-failure \
        --net=host \
        -d \
        gcr.io/google_containers/etcd-${ARCH}:${ETCD_VERSION} \
        /usr/local/bin/etcd \
            --listen-client-urls=http://127.0.0.1:4001,http://${MASTER_IP}:4001 \
            --advertise-client-urls=http://${MASTER_IP}:4001 \
            --data-dir=/var/etcd/data

    sleep 5
    # Set flannel net config
    docker -H unix:///var/run/docker-bootstrap.sock run \
        --net=host gcr.io/google_containers/etcd:${ETCD_VERSION} \
        etcdctl \
        set /coreos.com/network/config \
            '{ "Network": "10.1.0.0/16", "Backend": {"Type": "vxlan"}}'

    # iface may change to a private network interface, eth0 is for default
    flannelCID=$(docker -H unix:///var/run/docker-bootstrap.sock run \
        --restart=on-failure \
        -d \
        --net=host \
        --privileged \
        -v /dev/net:/dev/net \
        quay.io/coreos/flannel:${FLANNEL_VERSION} \
        /opt/bin/flanneld \
            --ip-masq="${FLANNEL_IPMASQ}" \
            --iface="${FLANNEL_IFACE}")

    sleep 8

    # Copy flannel env out and source it on the host
    docker -H unix:///var/run/docker-bootstrap.sock \
        cp ${flannelCID}:/run/flannel/subnet.env .
    source subnet.env

    # Configure docker net settings, then restart it
    case "${lsb_dist}" in
        amzn)
            DOCKER_CONF="/etc/sysconfig/docker"
            echo "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | tee -a ${DOCKER_CONF}
            ifconfig docker0 down
            yum -y -q install bridge-utils && brctl delbr docker0 && service docker restart
            ;;
        centos)
            DOCKER_CONF="/etc/sysconfig/docker"
            echo "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | tee -a ${DOCKER_CONF}
            if ! command_exists ifconfig; then
                yum -y -q install net-tools
            fi
            ifconfig docker0 down
            yum -y -q install bridge-utils && brctl delbr docker0 && systemctl restart docker
            ;;
        ubuntu|debian)
            DOCKER_CONF="/etc/default/docker"
            echo "DOCKER_OPTS=\"\$DOCKER_OPTS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | tee -a ${DOCKER_CONF}
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

    # sleep a little bit
    sleep 5

    # Start kubelet and then start master components as pods
    docker run \
        --net=host \
        --pid=host \
        --privileged \
        --restart=on-failure \
        -d \
        -v /sys:/sys:ro \
        -v /var/run:/var/run:rw \
        -v /:/rootfs:ro \
        -v /var/lib/docker/:/var/lib/docker:rw \
        -v /var/lib/kubelet/:/var/lib/kubelet:rw \
        gcr.io/google_containers/hyperkube-${ARCH}:v${K8S_VERSION} \
        /hyperkube kubelet \
            --address=0.0.0.0 \
            --allow-privileged=true \
            --enable-server \
            --api-servers=http://localhost:8080 \
            --config=/etc/kubernetes/manifests-multi \
            --cluster-dns=10.0.0.10 \
            --cluster-domain=cluster.local \
            --containerized \
            --v=2

}

echo "Detecting your OS distro ..."
detect_lsb

echo "Starting bootstrap docker ..."
bootstrap_daemon

echo "Starting k8s ..."
start_k8s

echo "Master done!"
