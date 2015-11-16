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

# A common lib for master.sh & node.sh to use
source ~/docker/kube-config/node.env
source ~/docker/kube-config/kubelet.env

DESTROY_SH=~/docker/kube-deploy/destroy.sh

: ${K8S_VERSION?"K8S_VERSION is not popultated on Node"}

# Run as root
if [ "$(id -u)" != "0" ]; then
    echo >&2 "Please run as root"
    exit 1
fi

# Make sure docker daemon is running
if ( ! ps -ef | grep "/usr/bin/docker" | grep -v 'grep' &> /dev/null ); then
    echo "Docker daemon is not running on this machine!"
    exit 1
fi

# Check if a command is valid
function command_exists() {
    command -v "$@" > /dev/null 2>&1
}

# Detect the OS distro, we support ubuntu, debian, mint, centos, fedora dist
function detect_lsb() {
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

    export lsb_dist="$(echo ${lsb_dist} | tr '[:upper:]' '[:lower:]')"
    echo "... ... OS distro Detected: $lsb_dist"

    case "${lsb_dist}" in
        amzn|centos|debian|ubuntu)
            ;;
        *)
            echo "... ... Error: We currently only support ubuntu|debian|amzn|centos."
            exit 1
            ;;
    esac

}


# Start the bootstrap daemon
function bootstrap_daemon() {
    local pid=`ps -eaf | grep 'unix:///var/run/docker-bootstrap.sock' | grep -v grep | awk '{print $2}'`

    if [[ -z "$pid" ]]; then
        sudo -b docker -d -H unix:///var/run/docker-bootstrap.sock \
            -p /var/run/docker-bootstrap.pid --iptables=false --ip-masq=false \
            --bridge=none --graph=/var/lib/docker-bootstrap \
            2> /var/log/docker-bootstrap.log 1> /dev/null

        # Wait for bootstrap daemon ready
        sleep 2
    else
        echo "... Bootstrap daemon already existed, try to clear its containers"
        $DESTROY_SH clear_bootstrap_containers >/dev/null 2>&1
    fi

    echo "... Bootstrap daemon started"
}

# Deploy network implementation for k8s
function start-network() {
  echo "... Configuring network"
  # $1 is a flag for config-network to know if it is deploying a master
  ~/docker/config-network.sh $1
}

