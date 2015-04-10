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

# A library of helper functions that each provider hosting Kubernetes must implement to use cluster/kube-*.sh scripts.
set -e

SSH_OPTS="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR"

# use an array to record name and ip
declare -A mm
CLUSTER=""
MASTER=""
MASTER_IP=""
MINION_IPS=""

# From user input set the necessary k8s and etcd configuration infomation
function setClusterInfo() {
  ii=0
  for i in $nodes
  do
      name="infra"$ii
      nodeIP=${i#*@}

      item="$name=http://$nodeIP:2380"
      if [ "$ii" == 0 ]; then 
          CLUSTER=$item
      else
          CLUSTER="$CLUSTER,$item"
      fi
      mm[$nodeIP]=$name

      if [ "${roles[${ii}]}" == "ai" ]; then
        MASTER_IP=$nodeIP
        MASTER=$i
        MINION_IPS="$nodeIP"
      elif [ "${roles[${ii}]}" == "a" ]; then 
        MASTER_IP=$nodeIP
        MASTER=$i
      elif [ "${roles[${ii}]}" == "i" ]; then
        if [ -z "${MINION_IPS}" ];then
          MINION_IPS="$nodeIP"
        else
          MINION_IPS="$MINION_IPS,$nodeIP"
        fi
      else
        echo "unsupported role for ${i}. please check"
        exit 1
      fi

      ((ii=ii+1))
  done

}


# Verify ssh prereqs
function verify-prereqs {
   # Expect at least one identity to be available.
  if ! ssh-add -L 1> /dev/null 2> /dev/null; then
    echo "Could not find or add an SSH identity."
    echo "Please start ssh-agent, add your identity, and retry."
    exit 1
  fi
}

# Check prereqs on every k8s node
function check-prereqs {
  PATH=$PATH:/opt/bin/
  # use ubuntu
  if ! $(grep Ubuntu /etc/lsb-release > /dev/null 2>&1)
  then
      echo "warning: not detecting a ubuntu system"
      exit 1
  fi

  # check etcd
  if ! $(which etcd > /dev/null)
  then
      echo "warning: etcd binary is not found in the PATH: $PATH"
      exit 1
  fi

  # detect the etcd version, we support only etcd 2.0.
  etcdVersion=$(/opt/bin/etcd --version | awk '{print $3}')

  if [ "$etcdVersion" != "2.0.0" ]; then
    echo "We only support 2.0.0 version of etcd"
    exit 1
  fi
}

function verify-cluster {
  ii=0

  for i in ${nodes}
  do
    if [ "${roles[${ii}]}" == "a" ]; then
      verify-master 
    elif [ "${roles[${ii}]}" == "i" ]; then
      verify-minion $i
    elif [ "${roles[${ii}]}" == "ai" ]; then
      verify-master
      verify-minion $i
    else
      echo "unsupported role for ${i}. please check"
      exit 1
    fi

    ((ii=ii+1))
  done

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  http://${MASTER_IP}"
  echo

}

function verify-master(){
  # verify master has all required daemons
  echo "Validating master"
  local -a required_daemon=("kube-apiserver" "kube-controller-manager" "kube-scheduler")
  local validated="1"
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
        ssh "$MASTER" "pgrep -f ${daemon}" >/dev/null 2>&1 || {
        printf "."
        validated="1"
        sleep 2
      }
    done
  done

}

function verify-minion(){
  # verify minion has all required daemons
  echo "Validating ${1}"
  local -a required_daemon=("kube-proxy" "kubelet" "docker")
  local validated="1"
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
        ssh "$1" "pgrep -f $daemon" >/dev/null 2>&1 || {
        printf "."
        validated="1"
        sleep 2
      }
    done
  done
}

function create-etcd-opts(){
  cat <<EOF > ~/kube/default/etcd
ETCD_OPTS="-name $1 \
  -initial-advertise-peer-urls http://$2:2380 \
  -listen-peer-urls http://$2:2380 \
  -initial-cluster-token etcd-cluster-1 \
  -initial-cluster $3 \
  -initial-cluster-state new"
EOF
}

function create-kube-apiserver-opts(){
  cat <<EOF > ~/kube/default/kube-apiserver
KUBE_APISERVER_OPTS="--address=0.0.0.0 \
--port=8080 \
--etcd_servers=http://127.0.0.1:4001 \
--logtostderr=true \
--portal_net=${1}"
EOF
}

function create-kube-controller-manager-opts(){
  cat <<EOF > ~/kube/default/kube-controller-manager
KUBE_CONTROLLER_MANAGER_OPTS="--master=127.0.0.1:8080 \
--machines=$1 \
--logtostderr=true"
EOF

}

function create-kube-scheduler-opts(){
  cat <<EOF > ~/kube/default/kube-scheduler
KUBE_SCHEDULER_OPTS="--logtostderr=true \
--master=127.0.0.1:8080"
EOF

}

function create-kubelet-opts(){
  cat <<EOF > ~/kube/default/kubelet
KUBELET_OPTS="--address=0.0.0.0 \
--port=10250 \
--hostname_override=$1 \
--api_servers=http://$2:8080 \
--logtostderr=true \
--cluster_dns=$3 \
--cluster_domain=$4"
EOF

}

function create-kube-proxy-opts(){
  cat <<EOF > ~/kube/default/kube-proxy
KUBE_PROXY_OPTS="--master=http://${1}:8080 \
--logtostderr=true"
EOF

}

function create-flanneld-opts(){
  cat <<EOF > ~/kube/default/flanneld
FLANNEL_OPTS=""
EOF
}

# Ensure that we have a password created for validating to the master. Will
# read from $HOME/.kubernetes_auth if available.
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function get-password {
  local file="$HOME/.kubernetes_auth"
  if [[ -r "$file" ]]; then
    KUBE_USER=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
    KUBE_PASSWORD=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
    return
  fi
  KUBE_USER=admin
  KUBE_PASSWORD=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')

  # Store password for reuse.
  cat << EOF > "$file"
{
  "User": "$KUBE_USER",
  "Password": "$KUBE_PASSWORD"
}
EOF
  chmod 0600 "$file"
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP
function detect-master {
  KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"
  setClusterInfo
  KUBE_MASTER=$MASTER
  KUBE_MASTER_IP=$MASTER_IP
  echo "Using master $MASTER_IP"
}

# Detect the information about the minions
#
# Assumed vars:
#   nodes
# Vars set:
#   KUBE_MINION_IP_ADDRESS (array)
function detect-minions {
  KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"

  KUBE_MINION_IP_ADDRESSES=()
  setClusterInfo
  
  ii=0
  for i in ${nodes}
  do
    if [ "${roles[${ii}]}" == "i" ] || [ "${roles[${ii}]}" == "ai" ]; then
      KUBE_MINION_IP_ADDRESSES+=("${i#*@}")
    fi

    ((ii=ii+1))
  done

  if [[ -z "${KUBE_MINION_IP_ADDRESSES[@]}" ]]; then
    echo "Could not detect Kubernetes minion nodes. Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Instantiate a kubernetes cluster on ubuntu
function kube-up {
  KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"

  # ensure the binaries are downloaded
  if [ ! -f "ubuntu/binaries/master/kube-apiserver" ]; then
    echo "warning: not enough binaries to build k8s, please run build.sh in cluster/ubuntu first"
    exit 1
  fi

  setClusterInfo
  ii=0

  for i in ${nodes}
  do
  {
    if [ "${roles[${ii}]}" == "a" ]; then
      provision-master 
    elif [ "${roles[${ii}]}" == "i" ]; then
      provision-minion $i 
    elif [ "${roles[${ii}]}" == "ai" ]; then
      provision-masterandminion
    else
      echo "unsupported role for ${i}. please check"
      exit 1
    fi
  }

    ((ii=ii+1))
        
  done
  wait

  verify-cluster
}

function provision-master() {
  # copy the binaries and scripts to the ~/kube directory on the master
  echo "Deploying master on machine ${MASTER_IP}"
  echo 
  ssh $SSH_OPTS $MASTER "mkdir -p ~/kube/default"
  scp -r $SSH_OPTS ubuntu/config-default.sh ubuntu/util.sh ubuntu/master/* ubuntu/binaries/master/ "${MASTER}:~/kube"

  # remote login to MASTER and use sudo to configue k8s master
  ssh $SSH_OPTS -t $MASTER "source ~/kube/util.sh; \
                            setClusterInfo; \
                            create-etcd-opts "${mm[${MASTER_IP}]}" "${MASTER_IP}" "${CLUSTER}"; \
                            create-kube-apiserver-opts "${PORTAL_NET}"; \
                            create-kube-controller-manager-opts "${MINION_IPS}"; \
                            create-kube-scheduler-opts; \
                            sudo -p '[sudo] password to copy files and start master: ' cp ~/kube/default/* /etc/default/ && sudo cp ~/kube/init_conf/* /etc/init/ && sudo cp ~/kube/init_scripts/* /etc/init.d/ \
                            && sudo mkdir -p /opt/bin/ && sudo cp ~/kube/master/* /opt/bin/; \
                            sudo service etcd start;"
}

function provision-minion() {
    # copy the binaries and scripts to the ~/kube directory on the minion
    echo "Deploying minion on machine ${1#*@}"
    echo
    ssh $SSH_OPTS $1 "mkdir -p ~/kube/default"
    scp -r $SSH_OPTS ubuntu/config-default.sh ubuntu/util.sh ubuntu/reconfDocker.sh ubuntu/minion/* ubuntu/binaries/minion "${1}:~/kube"

    # remote login to MASTER and use sudo to configue k8s master
    ssh $SSH_OPTS -t $1 "source ~/kube/util.sh; \
                         setClusterInfo; \
                         create-etcd-opts "${mm[${1#*@}]}" "${1#*@}" "${CLUSTER}"; \
                         create-kubelet-opts "${1#*@}" "${MASTER_IP}" "${DNS_SERVER_IP}" "${DNS_DOMAIN}"; 
                         create-kube-proxy-opts "${MASTER_IP}"; \
                         create-flanneld-opts; \
                         sudo -p '[sudo] password to copy files and start minion: ' cp ~/kube/default/* /etc/default/ && sudo cp ~/kube/init_conf/* /etc/init/ && sudo cp ~/kube/init_scripts/* /etc/init.d/ \
                         && sudo mkdir -p /opt/bin/ && sudo cp ~/kube/minion/* /opt/bin; \
                         sudo service etcd start; \
                         sudo -b ~/kube/reconfDocker.sh"
}

function provision-masterandminion() {
  # copy the binaries and scripts to the ~/kube directory on the master
  echo "Deploying master and minion on machine ${MASTER_IP}"
  echo 
  ssh $SSH_OPTS $MASTER "mkdir -p ~/kube/default"
  scp -r $SSH_OPTS ubuntu/config-default.sh ubuntu/util.sh ubuntu/master/* ubuntu/reconfDocker.sh ubuntu/minion/* ubuntu/binaries/master/ ubuntu/binaries/minion "${MASTER}:~/kube"
  
  # remote login to the node and use sudo to configue k8s
  ssh $SSH_OPTS -t $MASTER "source ~/kube/util.sh; \
                            setClusterInfo; \
                            create-etcd-opts "${mm[${MASTER_IP}]}" "${MASTER_IP}" "${CLUSTER}"; \
                            create-kube-apiserver-opts "${PORTAL_NET}"; \
                            create-kube-controller-manager-opts "${MINION_IPS}"; \
                            create-kube-scheduler-opts; \
                            create-kubelet-opts "${MASTER_IP}" "${MASTER_IP}" "${DNS_SERVER_IP}" "${DNS_DOMAIN}";                     
                            create-kube-proxy-opts "${MASTER_IP}";\
                            create-flanneld-opts; \
                            sudo -p '[sudo] password to copy files and start node: ' cp ~/kube/default/* /etc/default/ && sudo cp ~/kube/init_conf/* /etc/init/ && sudo cp ~/kube/init_scripts/* /etc/init.d/ \
                            && sudo mkdir -p /opt/bin/ && sudo cp ~/kube/master/* /opt/bin/ && sudo cp ~/kube/minion/* /opt/bin/; \
                            sudo service etcd start; \
                            sudo -b ~/kube/reconfDocker.sh"
}

# Delete a kubernetes cluster
function kube-down {
  KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"

  for i in ${nodes}; do
  {
    echo "Cleaning on node ${i#*@}"
    ssh -t $i 'pgrep etcd && sudo -p "[sudo] password for cleaning etcd data: " service etcd stop && sudo rm -rf /infra*'
  } 
  done
  wait
}

# Update a kubernetes cluster with latest source
function kube-push {
  echo "not implemented"
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  echo "Ubuntu doesn't need special preparations for e2e tests" 1>&2
}