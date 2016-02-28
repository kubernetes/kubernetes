#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# Utility functions for Kubernetes in docker setup
# Authors @luxas @wizard_cxy @resouer @loopingz 

# Variables
K8S_VERSION=${K8S_VERSION:-"1.2.0-alpha.8"}
ETCD_VERSION=${ETCD_VERSION:-"2.2.1"}
FLANNEL_VERSION=${FLANNEL_VERSION:-"0.5.5"}
FLANNEL_IFACE=${FLANNEL_IFACE:-"eth0"}
FLANNEL_IPMASQ=${FLANNEL_IPMASQ:-"true"}
FLANNEL_BACKEND=${FLANNEL_BACKEND:-"vxlan"}
FLANNEL_NETWORK=${FLANNEL_NETWORK:-"10.1.0.0/16"}
DNS_DOMAIN=${DNS_DOMAIN:-"cluster.local"}
DNS_SERVER_IP=${DNS_SERVER_IP:-"10.0.0.10"}
RESTART_POLICY=${RESTART_POLICY:-"on-failure"}
ARCH=${ARCH:-"amd64"}

# Constants
TIMEOUT_FOR_SERVICES=20
BOOTSTRAP_DOCKER_SOCK="unix:///var/run/docker-bootstrap.sock"
KUBELET_MOUNTS="\
  -v /sys:/sys:ro \
  -v /var/run:/var/run:rw \
  -v /:/rootfs:ro \
  -v /var/lib/docker/:/var/lib/docker:rw \
  -v /var/lib/kubelet/:/var/lib/kubelet:rw"

# Paths
FLANNEL_SUBNET_TMPDIR=$(mktemp -d)
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../..

# Source useful scripts
source "${KUBE_ROOT}/hack/lib/util.sh"
source "${KUBE_ROOT}/cluster/lib/logging.sh"

# Trap errors
kube::log::install_errexit

# Ensure everything is OK, docker is running and we're root
kube::multinode::check_params() {

  # Make sure docker daemon is running
  if [[ $(docker ps 2>&1 1>/dev/null; echo $?) != 0 ]]; then
    kube::log::error "Docker is not running on this machine!"
    exit 1
  fi

  # Require root
  if [[ "$(id -u)" != "0" ]]; then
    kube::log::error >&2 "Please run as root"
    exit 1
  fi

  # Output the value of the variables
  kube::log::status "K8S_VERSION is set to: ${K8S_VERSION}"
  kube::log::status "ETCD_VERSION is set to: ${ETCD_VERSION}"
  kube::log::status "FLANNEL_VERSION is set to: ${FLANNEL_VERSION}"
  kube::log::status "FLANNEL_IFACE is set to: ${FLANNEL_IFACE}"
  kube::log::status "FLANNEL_IPMASQ is set to: ${FLANNEL_IPMASQ}"
  kube::log::status "FLANNEL_NETWORK is set to: ${FLANNEL_NETWORK}"
  kube::log::status "FLANNEL_BACKEND is set to: ${FLANNEL_BACKEND}"
  kube::log::status "DNS_DOMAIN is set to: ${DNS_DOMAIN}"
  kube::log::status "DNS_SERVER_IP is set to: ${DNS_SERVER_IP}"
  kube::log::status "RESTART_POLICY is set to: ${RESTART_POLICY}"
  kube::log::status "MASTER_IP is set to: ${MASTER_IP}"
  kube::log::status "ARCH is set to: ${ARCH}"
  kube::log::status "--------------------------------------------"
}

# Detect the OS distro, we support ubuntu, debian, mint, centos, fedora and systemd dist
kube::multinode::detect_lsb() {

  # TODO(luxas): add support for arm, arm64 and ppc64le
  case "$(kube::util::host_platform)" in
    linux/amd64)
      ;;
    *)
      kube::log::error "Error: We currently only support the linux/amd64 platform."
      exit 1
      ;;
  esac

  if kube::helpers::command_exists lsb_release; then
    lsb_dist="$(lsb_release -si)"
  elif [[ -r /etc/lsb-release ]]; then
    lsb_dist="$(. /etc/lsb-release && echo "$DISTRIB_ID")"
  elif [[ -r /etc/debian_version ]]; then
    lsb_dist='debian'
  elif [[ -r /etc/fedora-release ]]; then
    lsb_dist='fedora'
  elif [[ -r /etc/os-release ]]; then
    lsb_dist="$(. /etc/os-release && echo "$ID")"
  elif kube::helpers::command_exists systemctl; then
    lsb_dist='systemd'
  fi

  lsb_dist="$(echo ${lsb_dist} | tr '[:upper:]' '[:lower:]')"

  case "${lsb_dist}" in
      amzn|centos|debian|ubuntu|systemd)
          ;;
      *)
          kube::log::error "Error: We currently only support ubuntu|debian|amzn|centos|systemd."
          exit 1
          ;;
  esac

  kube::log::status "Detected OS: ${lsb_dist}"
}

# Start a docker bootstrap for running etcd and flannel
kube::multinode::bootstrap_daemon() {

  kube::log::status "Launching docker bootstrap..."

  docker daemon \
    -H ${BOOTSTRAP_DOCKER_SOCK} \
    -p /var/run/docker-bootstrap.pid \
    --iptables=false \
    --ip-masq=false \
    --bridge=none \
    --graph=/var/lib/docker-bootstrap \
      2> /var/log/docker-bootstrap.log \
      1> /dev/null &

  # Wait for docker bootstrap to start by "docker ps"-ing every second
  local BOOTSTRAP_SECONDS=0
  while [[ $(docker -H ${BOOTSTRAP_DOCKER_SOCK} ps 2>&1 1>/dev/null; echo $?) != 0 ]]; do
    ((BOOTSTRAP_SECONDS++))
    if [[ ${BOOTSTRAP_SECONDS} == ${TIMEOUT_FOR_SERVICES} ]]; then
      kube::log::error "docker bootstrap failed to start. Exiting..."
      exit
    fi
    sleep 1
  done
}

# Start etcd on the master node
kube::multinode::start_etcd() {

  kube::log::status "Launching etcd..."
  
  docker -H ${BOOTSTRAP_DOCKER_SOCK} run -d \
    --restart=${RESTART_POLICY} \
    --net=host \
    gcr.io/google_containers/etcd-${ARCH}:${ETCD_VERSION} \
    /usr/local/bin/etcd \
      --listen-client-urls=http://127.0.0.1:4001,http://${MASTER_IP}:4001 \
      --advertise-client-urls=http://${MASTER_IP}:4001 \
      --data-dir=/var/etcd/data

  # Wait for etcd to come up
  kube::util::wait_for_url "http://localhost:4001/v2/machines" "etcd" 0.25 80

  # Set flannel net config
  docker -H ${BOOTSTRAP_DOCKER_SOCK} run \
      --net=host \
      gcr.io/google_containers/etcd-${ARCH}:${ETCD_VERSION} \
      etcdctl \
      set /coreos.com/network/config \
          "{ \"Network\": \"${FLANNEL_NETWORK}\", \"Backend\": {\"Type\": \"${FLANNEL_BACKEND}\"}}"

  sleep 2
}

# Start flannel in docker bootstrap, both for master and worker
kube::multinode::start_flannel() {

  kube::log::status "Launching flannel..."

  docker -H ${BOOTSTRAP_DOCKER_SOCK} run \
    --restart=${RESTART_POLICY} \
    -d \
    --net=host \
    --privileged \
    -v /dev/net:/dev/net \
    -v ${FLANNEL_SUBNET_TMPDIR}:/run/flannel \
    quay.io/coreos/flannel:${FLANNEL_VERSION} \
    /opt/bin/flanneld \
      --etcd-endpoints=http://${MASTER_IP}:4001 \
      --ip-masq="${FLANNEL_IPMASQ}" \
      --iface="${FLANNEL_IFACE}"

  # Wait for the flannel subnet.env file to be created instead of a timeout. This is faster and more reliable
  local FLANNEL_SECONDS=0
  while [[ ! -f ${FLANNEL_SUBNET_TMPDIR}/subnet.env ]]; do
    ((FLANNEL_SECONDS++))
    if [[ ${FLANNEL_SECONDS} == 20 ]]; then
      kube::log::error "flannel failed to start. Exiting..."
      exit
    fi
    sleep 1
  done

  source ${FLANNEL_SUBNET_TMPDIR}/subnet.env

  kube::log::status "FLANNEL_SUBNET is set to: ${FLANNEL_SUBNET}"
  kube::log::status "FLANNEL_MTU is set to: ${FLANNEL_MTU}"
}

# Configure docker net settings, then restart it
kube::multinode::restart_docker(){

  case "${lsb_dist}" in
    amzn)
      DOCKER_CONF="/etc/sysconfig/docker"

      kube::helpers::file_replace_line ${DOCKER_CONF} \ # Replace content in this file
        "--bip" \ # Find a line with this content...
        "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" # ...and replace the found line with this line

      ifconfig docker0 down
      yum -y -q install bridge-utils 
      brctl delbr docker0 
      service docker restart
      ;;
    centos)
      DOCKER_CONF="/etc/sysconfig/docker"

      kube::helpers::file_replace_line ${DOCKER_CONF} \ # Replace content in this file
        "--bip" \ # Find a line with this content...
        "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" # ...and replace the found line with this line
      
      if ! kube::helpers::command_exists ifconfig; then
          yum -y -q install net-tools
      fi

      yum -y -q install bridge-utils
      ifconfig docker0 down
      brctl delbr docker0 
      systemctl restart docker
      ;;
    ubuntu|debian)
      # Newer ubuntu and debian releases uses systemd. Handle that
      if kube::helpers::command_exists systemctl; then
        kube::multinode::restart_docker_systemd
      else
        DOCKER_CONF="/etc/default/docker"
        
        kube::helpers::file_replace_line ${DOCKER_CONF} \ # Replace content in this file
          "--bip" \ # Find a line with this content...
          "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" # ...and replace the found line with this line

        apt-get install -y bridge-utils 
        brctl delbr docker0 
        service docker stop
        while [ $(ps aux | grep /usr/bin/docker | grep -v grep | wc -l) -gt 0 ]; do
            kube::log::status "Waiting for docker to terminate"
            sleep 1
        done
        service docker start
      fi
      ;;
    systemd)
      kube::multinode::restart_docker_systemd
      ;;
    *)
        kube::log::error "Unsupported operations system ${lsb_dist}"
        exit 1
        ;;
  esac

  kube::log::status "Restarted docker with the new flannel settings"
}

# Replace --mtu and --bip in systemd's docker.service file and restart
kube::multinode::restart_docker_systemd(){

  DOCKER_CONF="/lib/systemd/system/docker.service"

  # This expression checks if the "--mtu" and "--bip" options are there
  # If they aren't, they are inserted at the end of the docker command
  if [[ -z $(grep -- "--mtu=" $DOCKER_CONF) ]]; then
    sed -e "s@$(grep "/usr/bin/docker" $DOCKER_CONF)@$(grep "/usr/bin/docker" $DOCKER_CONF) --mtu=${FLANNEL_MTU}@g" -i $DOCKER_CONF
  fi
  if [[ -z $(grep -- "--bip=" $DOCKER_CONF) ]]; then
    sed -e "s@$(grep "/usr/bin/docker" $DOCKER_CONF)@$(grep "/usr/bin/docker" $DOCKER_CONF) --bip=${FLANNEL_SUBNET}@g" -i $DOCKER_CONF
  fi

  # Finds "--mtu=????" and replaces with "--mtu=${FLANNEL_MTU}"
  # Also finds "--bip=??.??.??.??" and replaces with "--bip=${FLANNEL_SUBNET}"
  sed -e "s@$(grep -o -- "--mtu=[[:graph:]]*" $DOCKER_CONF)@--mtu=${FLANNEL_MTU}@g;s@$(grep -o -- "--bip=[[:graph:]]*" $DOCKER_CONF)@--bip=${FLANNEL_SUBNET}@g" -i $DOCKER_CONF

  systemctl daemon-reload
  systemctl restart docker
}

# Start kubelet first and then the master components as pods
kube::multinode::start_k8s_master() {
  
  kube::log::status "Launching Kubernetes master components..."

  docker run -d \
    --net=host \
    --pid=host \
    --privileged \
    --restart=${RESTART_POLICY} \
    ${KUBELET_MOUNTS} \
    gcr.io/google_containers/hyperkube-${ARCH}:v${K8S_VERSION} \
    /hyperkube kubelet \
      --allow-privileged=true \
      --api-servers=http://localhost:8080 \
      --config=/etc/kubernetes/manifests-multi \
      --cluster-dns=${DNS_SERVER_IP} \
      --cluster-domain=${DNS_DOMAIN} \
      --containerized \
      --v=2
}

# Start kubelet in a container, for a worker node
kube::multinode::start_k8s_worker() {
  
  kube::log::status "Launching Kubernetes worker components..."

  # TODO: Use secure port for communication
  docker run -d \
    --net=host \
    --pid=host \
    --privileged \
    --restart=${RESTART_POLICY} \
    ${KUBELET_MOUNTS} \
    gcr.io/google_containers/hyperkube-${ARCH}:v${K8S_VERSION} \
    /hyperkube kubelet \
      --allow-privileged=true \
      --api-servers=http://${MASTER_IP}:8080 \
      --cluster-dns=${DNS_SERVER_IP} \
      --cluster-domain=${DNS_DOMAIN} \
      --containerized \
      --v=2
}

# Start kube-proxy in a container, for a worker node
kube::multinode::start_k8s_worker_proxy() {

  kube::log::status "Launching kube-proxy..."
  docker run -d \
    --net=host \
    --privileged \
    --restart=${RESTART_POLICY} \
    gcr.io/google_containers/hyperkube-${ARCH}:v${K8S_VERSION} \
    /hyperkube proxy \
        --master=http://${MASTER_IP}:8080 \
        --v=2
}

# Turndown the local cluster
kube::multinode::turndown(){

  # Check if docker bootstrap is running
  if [[ $(kube::helpers::is_running ${BOOTSTRAP_DOCKER_SOCK}) == "true" ]]; then

    kube::log::status "Killing docker bootstrap..."

    # Kill all docker bootstrap's containers
    if [[ $(docker -H ${BOOTSTRAP_DOCKER_SOCK} ps -q | wc -l) != 0 ]]; then
      docker -H ${BOOTSTRAP_DOCKER_SOCK} rm -f $(docker -H ${BOOTSTRAP_DOCKER_SOCK} ps -q)
    fi

    # Kill bootstrap docker
    kill $(ps aux | grep ${BOOTSTRAP_DOCKER_SOCK} | grep -v grep | awk '{print $2}')

  fi

  if [[ $(kube::helpers::is_running /hyperkube) == "true" ]]; then
    
    kube::log::status "Killing hyperkube containers..."

    # Kill all hyperkube docker images
    docker rm -f $(docker ps | grep gcr.io/google_containers/hyperkube | awk '{print $1}')
  fi

  if [[ $(kube::helpers::is_running /pause) == "true" ]]; then
    
    kube::log::status "Killing pause containers..."

    # Kill all pause docker images
    docker rm -f $(docker ps | grep gcr.io/google_containers/pause | awk '{print $1}')
  fi

  if [[ -d /var/lib/kubelet ]]; then
    read -p "Do you want to clean /var/lib/kubelet? [Y/n] " clean_kubelet_dir

    case $clean_kubelet_dir in
      [n|N]*)
        ;; # Do nothing
      *)
        # umount if there's mounts bound in /var/lib/kubelet
        if [[ ! -z $(mount | grep /var/lib/kubelet | awk '{print $3}') ]]; then
          umount $(mount | grep /var/lib/kubelet | awk '{print $3}')
        fi

        # Delete the directory
        rm -rf /var/lib/kubelet
    esac
  fi
}

## Helpers

# Check if a command is valid
kube::helpers::command_exists() {
    command -v "$@" > /dev/null 2>&1
}

# Usage: kube::helpers::file_replace_line {path_to_file} {value_to_search_for} {replace_that_line_with_this_content}
# Finds a line in a file and replaces the line with the third argument
kube::helpers::file_replace_line(){
  if [[ -z $(grep "$2" $1) ]]; then
    echo "$3" >> $1
  else
    sed -i "/$2/c\\$3" $1
  fi
}

# Check if a process is running
kube::helpers::is_running(){
  if [[ ! -z $(ps aux | grep ${1} | grep -v grep) ]]; then
    echo "true"
  else
    echo "false"
  fi
}