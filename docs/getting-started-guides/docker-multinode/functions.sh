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

# Utility function for kubernetes docker setup
# Author @wizard_cxy @reouser @loopingz @luxas

K8S_VERSION=${K8S_VERSION:-"1.2.0-alpha.6"}
ETCD_VERSION=${ETCD_VERSION:-"2.2.1"}
FLANNEL_VERSION=${FLANNEL_VERSION:-"0.5.5"}
FLANNEL_IFACE=${FLANNEL_IFACE:-"eth0"}
ARCH=${ARCH:-"amd64"}

bootstrap_daemon() {
    RUNNING=`ps -efa|grep docker-bootstrap.sock|wc -l`
    if [ "$RUNNING" -gt "1" ]; then
        echo "Bootstrap is already running"
        return
    fi
    sudo -b docker -d \
    -H unix:///var/run/docker-bootstrap.sock \
    -p /var/run/docker-bootstrap.pid \
    --iptables=false \
    --ip-masq=false \
    --bridge=none \
    --graph=/var/lib/docker-bootstrap \
    2> /var/log/docker-bootstrap.log \
    1> /dev/null

    sleep 5
}

local_master() {
  RESULT=`ifconfig | grep $MASTER_IP | wc -l`
  if [ "$RESULT" -eq 0 ]; then
    return 1
  else
     return 0
  fi
}
# Check if a command is valid
command_exists() {
    command -v "$@" > /dev/null 2>&1
}

docker_run() {
  RUNNING=`docker ps|grep "$@"|wc -l`
  if [ "$RUNNING" -eq 0 ]; then
    return 1
  else
    return 0
  fi
}

docker_bootstrap_run() {
  RUNNING=`docker-bootstrap ps|grep "$@"|wc -l`
  if [ "$RUNNING" -eq 0 ]; then
    return 1
  else
    return 0
  fi
}

check_params() {
    # Make sure docker daemon is running
    if ( ! ps -ef | grep "/usr/bin/docker" | grep -v 'grep' &> /dev/null ); then
      echo "Docker is not running on this machine!"
      exit 1
    fi
    # Run as root
    if [ "$(id -u)" != "0" ]; then
      echo >&2 "Please run as root"
      exit 1
    fi
    # Get current ip
    CURRENT_IP=`ifconfig $FLANNEL_IF | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'`
    if [ -z ${CURRENT_IP} ]; then
      CURRENT_IP=`hostname -i|xargs`
    fi
    # Make sure master ip is properly set
    if [ -z ${MASTER_IP} ]; then
        echo "Please export MASTER_IP in your env"
        exit 1
    fi
    echo "K8S_VERSION is set to: ${K8S_VERSION}"
    echo "ETCD_VERSION is set to: ${ETCD_VERSION}"
    echo "FLANNEL_VERSION is set to: ${FLANNEL_VERSION}"
    echo "FLANNEL_IFACE is set to: ${FLANNEL_IFACE}"
    echo "MASTER_IP is set to: ${MASTER_IP}"
    echo "ARCH is set to: ${ARCH}"
}

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

    case "${lsb_dist}" in
        amzn|centos|debian|ubuntu)
            ;;
        *)
            echo "Error: We currently only support ubuntu|debian|amzn|centos."
            exit 1
            ;;
    esac
}

etcd() {
  if docker_bootstrap_run etcd; then
    return 0
  fi
  if local_master; then
    echo "Launch new etcd cluster"
    etcd_master
  else
    etcd_join
  fi
}

etcd_master() {
   docker-bootstrap run \
      --restart=always \
      --net=host \
      --name=etcd \
      -d \
      gcr.io/google_containers/etcd:${ETCD_VERSION} \
      /usr/local/bin/etcd \
      --listen-client-urls=http://0.0.0.0:2379,http://0.0.0.0:4001 \
      --advertise-client-urls=http://${MASTER_IP}:2379,http://${MASTER_IP}:4001 \
      --data-dir=/var/etcd/data \
      --initial-advertise-peer-urls http://${MASTER_IP}:2380 \
      --listen-peer-urls http://0.0.0.0:2380 \
      --initial-cluster-token `hostname` \
      --initial-cluster default=http://${MASTER_IP}:2380 \
      --initial-cluster-state new

      sleep 5
      # Set flannel net config
      docker -H unix:///var/run/docker-bootstrap.sock run \
      --net=host gcr.io/google_containers/etcd:${ETCD_VERSION} \
      etcdctl \
      set /coreos.com/network/config \
      '{ "Network": "10.1.0.0/16", "Backend": {"Type": "vxlan"}}'
}

etcd_join() {
  if [ ! -e "/etc/etcd-cluster.conf" ]; then
    echo "Joining etcd cluster"
    etcdctl --endpoint http://$MASTER_IP:4001 member add $NODE_NAME http://$CURRENT_IP:2380 | tail -n 3 > /etc/etcd-cluster.conf
  fi
  # Load configuration
  source /etc/etcd-cluster.conf
  echo "Will load etcd with"
  echo " - ETCD_NAME=$ETCD_NAME"
  echo " - ETCD_INITIAL_CLUSTER=$ETCD_INITIAL_CLUSTER"
  echo " - ETCD_INITIAL_CLUSTER_STATE=$ETCD_INITIAL_CLUSTER_STATE"
  docker-bootstrap run --restart=always --net=host --name=etcd -d gcr.io/google_containers/etcd:${ETCD_VERSION} /usr/local/bin/etcd \
           --listen-client-urls=http://0.0.0.0:2379,http://0.0.0.0:4001 \
           --advertise-client-urls=http://$CURRENT_IP:2379,http://$CURRENT_IP:4001 \
           --data-dir=/var/etcd/data \
           --listen-peer-urls http://0.0.0.0:2380 \
           --name $NODE_NAME \
           --initial-advertise-peer-urls http://$CURRENT_IP:2380 \
           --initial-cluster-token $ETCD_NAME --initial-cluster $ETCD_INITIAL_CLUSTER --initial-cluster-state $ETCD_INITIAL_CLUSTER_STATE
}

flannel() {
  if docker_bootstrap_run flannel; then
    echo "Flannel is already running"
    return 0
  fi
  echo "Launching flannel"
  flannelCID=$(docker -H unix:///var/run/docker-bootstrap.sock run \
      --restart=always \
      -d \
      --net=host \
      --name=flannel \
      --privileged \
      -v /dev/net:/dev/net \
      quay.io/coreos/flannel:${FLANNEL_VERSION} \
      /opt/bin/flanneld \
      --ip-masq \
      -iface="${FLANNEL_IF}")

      sleep 8
      # Copy flannel env out and source it on the host
      docker -H unix:///var/run/docker-bootstrap.sock \
          cp ${flannelCID}:/run/flannel/subnet.env .
      source subnet.env
      detect_lsb
      # Configure docker net settings, then restart it
      case "${lsb_dist}" in
          amzn)
              DOCKER_CONF="/etc/sysconfig/docker"
              echo "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | sudo tee -a ${DOCKER_CONF}
              ifconfig docker0 down
              yum -y -q install bridge-utils && brctl delbr docker0 && service docker restart
              ;;
          centos)
              DOCKER_CONF="/etc/sysconfig/docker"
              echo "OPTIONS=\"\$OPTIONS --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | sudo tee -a ${DOCKER_CONF}
              if ! command_exists ifconfig; then
                  yum -y -q install net-tools
              fi
              ifconfig docker0 down
              yum -y -q install bridge-utils && brctl delbr docker0 && systemctl restart docker
              ;;
          ubuntu|debian)
              DOCKER_CONF="/etc/default/docker"
              echo "DOCKER_OPTS=\" --dns=8.8.8.8 --mtu=${FLANNEL_MTU} --bip=${FLANNEL_SUBNET}\"" | sudo tee -a ${DOCKER_CONF}
              apt-get install bridge-utils
              ifconfig docker0 down
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
     sleep 5
}

kube() {
  if local_master; then
    k8s_master
  else
    k8s_worker
  fi
  k8s_proxy
}

k8s_worker() {
  if docker_run k8s-worker; then
    return 0
  fi
  echo "Launch k8s_worker"
  docker run \
          --net=host \
          --pid=host \
          --privileged \
          --restart=always \
          --name=k8s-worker \
          -d \
          -v /sys:/sys:ro \
          -v /var/run:/var/run:rw \
          -v /:/rootfs:ro \
          -v /dev:/dev \
          -v /var/lib/docker/:/var/lib/docker:rw \
          -v /var/lib/kubelet/:/var/lib/kubelet:rw \
          gcr.io/google_containers/hyperkube-${ARCH}:v${K8S_VERSION} \
          /hyperkube kubelet \
          --v=2 --address=0.0.0.0 --enable-server \
          --config=/etc/kubernetes/manifests-multi \
          --cluster-dns=10.0.0.10 \
          --cluster-domain=cluster.local \
          --api-servers=http://$MASTER_IP:8080 \
          --containerized
}

k8s_master() {
  if docker_run k8s-master; then
    return 0
  fi
  echo "Launch k8s_master"
  docker run \
          --net=host \
          --pid=host \
          --privileged \
          --restart=always \
          --name=k8s-master \
          -d \
          -v /sys:/sys:ro \
          -v /var/run:/var/run:rw \
          -v /:/rootfs:ro \
          -v /dev:/dev \
          -v /var/lib/docker/:/var/lib/docker:rw \
          -v /var/lib/kubelet/:/var/lib/kubelet:rw \
          gcr.io/google_containers/hyperkube-${ARCH}:v${K8S_VERSION} \
          /hyperkube kubelet \
          --v=2 --address=0.0.0.0 --enable-server \
          --config=/etc/kubernetes/manifests-multi \
          --cluster-dns=10.0.0.10 \
          --cluster-domain=cluster.local \
          --hostname-override=$MASTER_IP \
          --api-servers=http://$MASTER_IP:8080 \
          --containerized
}

k8s_proxy() {
  if docker_run k8s-proxy; then
    return 0
  fi
  echo "Launch k8s_proxy"
  docker run \
        -d \
        --net=host \
        --name=k8s-proxy \
        --privileged \
        gcr.io/google_containers/hyperkube-${ARCH}:v${K8S_VERSION} \
        /hyperkube proxy --master=http://$MASTER_IP:8080 --v=2  
}
