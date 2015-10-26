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

MASTER=""
MASTER_IP=""
NODE_IPS=""

# Assumed Vars:
#   KUBE_ROOT
function test-build-release {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# From user input set the necessary k8s and etcd configuration information
function setClusterInfo() {
  # Initialize NODE_IPS in setClusterInfo function
  # NODE_IPS is defined as a global variable, and is concatenated with other nodeIP	
  # When setClusterInfo is called for many times, this could cause potential problems
  # Such as, you will have NODE_IPS=192.168.0.2,192.168.0.3,192.168.0.2,192.168.0.3 which is obviously wrong
  NODE_IPS=""
  
  ii=0
  for i in $nodes; do
    nodeIP=${i#*@}

    if [[ "${roles[${ii}]}" == "ai" ]]; then
      MASTER_IP=$nodeIP
      MASTER=$i
      NODE_IPS="$nodeIP"
    elif [[ "${roles[${ii}]}" == "a" ]]; then
      MASTER_IP=$nodeIP
      MASTER=$i
    elif [[ "${roles[${ii}]}" == "i" ]]; then
      if [[ -z "${NODE_IPS}" ]];then
        NODE_IPS="$nodeIP"
      else
        NODE_IPS="$NODE_IPS,$nodeIP"
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
  local rc

  rc=0
  ssh-add -L 1> /dev/null 2> /dev/null || rc="$?"
  # "Could not open a connection to your authentication agent."
  if [[ "${rc}" -eq 2 ]]; then
    eval "$(ssh-agent)" > /dev/null
    trap-add "kill ${SSH_AGENT_PID}" EXIT
  fi

  rc=0
  ssh-add -L 1> /dev/null 2> /dev/null || rc="$?"
  # "The agent has no identities."
  if [[ "${rc}" -eq 1 ]]; then
    # Try adding one of the default identities, with or without passphrase.
    ssh-add || true
  fi
  # Expect at least one identity to be available.
  if ! ssh-add -L 1> /dev/null 2> /dev/null; then
    echo "Could not find or add an SSH identity."
    echo "Please start ssh-agent, add your identity, and retry."
    exit 1
  fi
}

# Install handler for signal trap
function trap-add {
  local handler="$1"
  local signal="${2-EXIT}"
  local cur

  cur="$(eval "sh -c 'echo \$3' -- $(trap -p ${signal})")"
  if [[ -n "${cur}" ]]; then
    handler="${cur}; ${handler}"
  fi

  trap "${handler}" ${signal}
}

function verify-cluster {
  ii=0

  for i in ${nodes}
  do
    if [ "${roles[${ii}]}" == "a" ]; then
      verify-master
    elif [ "${roles[${ii}]}" == "i" ]; then
      verify-node $i
    elif [ "${roles[${ii}]}" == "ai" ]; then
      verify-master
      verify-node $i
    else
      echo "unsupported role for ${i}. please check"
      exit 1
    fi

    ((ii=ii+1))
  done

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  http://${MASTER_IP}:8080"
  echo

}

function verify-master(){
  # verify master has all required daemons
  printf "Validating master"
  local -a required_daemon=("kube-apiserver" "kube-controller-manager" "kube-scheduler")
  local validated="1"
  local try_count=1
  local max_try_count=30
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
      ssh $SSH_OPTS "$MASTER" "pgrep -f ${daemon}" >/dev/null 2>&1 || {
        printf "."
        validated="1"
        ((try_count=try_count+1))
        if [[ ${try_count} -gt ${max_try_count} ]]; then
          printf "\nWarning: Process \"${daemon}\" failed to run on ${MASTER}, please check.\n"
          exit 1
        fi
        sleep 2
      }
    done
  done
  printf "\n"

}

function verify-node(){
  # verify node has all required daemons
  printf "Validating ${1}"
  local -a required_daemon=("kube-proxy" "kubelet" "docker")
  local validated="1"
  local try_count=1
  local max_try_count=30
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
      ssh $SSH_OPTS "$1" "pgrep -f $daemon" >/dev/null 2>&1 || {
        printf "."
        validated="1"
        ((try_count=try_count+1))
        if [[ ${try_count} -gt ${max_try_count} ]]; then
          printf "\nWarning: Process \"${daemon}\" failed to run on ${1}, please check.\n"
          exit 1
        fi
        sleep 2
      }
    done
  done
  printf "\n"
}

function create-etcd-opts(){
  cat <<EOF > ~/kube/default/etcd
ETCD_OPTS="-name infra
-listen-client-urls http://0.0.0.0:4001 \
-advertise-client-urls http://127.0.0.1:4001"
EOF
}

function create-kube-apiserver-opts(){
  cat <<EOF > ~/kube/default/kube-apiserver
KUBE_APISERVER_OPTS="--insecure-bind-address=0.0.0.0 \
--insecure-port=8080 \
--etcd-servers=http://127.0.0.1:4001 \
--logtostderr=true \
--service-cluster-ip-range=${1} \
--admission-control=${2} \
--service-node-port-range=${3} \
--client-ca-file=/srv/kubernetes/ca.crt \
--tls-cert-file=/srv/kubernetes/server.cert \
--tls-private-key-file=/srv/kubernetes/server.key"
EOF
}

function create-kube-controller-manager-opts(){
  cat <<EOF > ~/kube/default/kube-controller-manager
KUBE_CONTROLLER_MANAGER_OPTS="--master=127.0.0.1:8080 \
--root-ca-file=/srv/kubernetes/ca.crt \
--service-account-private-key-file=/srv/kubernetes/server.key \
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
--hostname-override=$1 \
--api-servers=http://$2:8080 \
--logtostderr=true \
--cluster-dns=$3 \
--cluster-domain=$4"
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
FLANNEL_OPTS="--etcd-endpoints=http://${1}:4001"
EOF
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP
function detect-master {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"
  setClusterInfo
  KUBE_MASTER=$MASTER
  KUBE_MASTER_IP=$MASTER_IP
  echo "Using master $MASTER_IP"
}

# Detect the information about the nodes
#
# Assumed vars:
#   nodes
# Vars set:
#   KUBE_NODE_IP_ADDRESS (array)
function detect-nodes {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"

  KUBE_NODE_IP_ADDRESSES=()
  setClusterInfo

  ii=0
  for i in ${nodes}
  do
    if [ "${roles[${ii}]}" == "i" ] || [ "${roles[${ii}]}" == "ai" ]; then
      KUBE_NODE_IP_ADDRESSES+=("${i#*@}")
    fi

    ((ii=ii+1))
  done

  if [[ -z "${KUBE_NODE_IP_ADDRESSES[@]}" ]]; then
    echo "Could not detect Kubernetes node nodes. Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Instantiate a kubernetes cluster on ubuntu
function kube-up() {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"

  # ensure the binaries are well prepared
  if [ ! -f "ubuntu/binaries/master/kube-apiserver" ]; then
    echo "No local binaries for kube-up, downloading... "
    "${KUBE_ROOT}/cluster/ubuntu/build.sh"
  fi

  setClusterInfo
  ii=0

  for i in ${nodes}
  do
    {
      if [ "${roles[${ii}]}" == "a" ]; then
        provision-master
      elif [ "${roles[${ii}]}" == "ai" ]; then
        provision-masterandnode
      elif [ "${roles[${ii}]}" == "i" ]; then
        provision-node $i
      else
        echo "unsupported role for ${i}. please check"
        exit 1
      fi
    }

    ((ii=ii+1))
  done
  wait

  verify-cluster
  detect-master
  export CONTEXT="ubuntu"
  export KUBE_SERVER="http://${KUBE_MASTER_IP}:8080"

  source "${KUBE_ROOT}/cluster/common.sh"

  # set kubernetes user and password
  load-or-gen-kube-basicauth

  create-kubeconfig
}

function provision-master() {
  # copy the binaries and scripts to the ~/kube directory on the master
  echo "Deploying master on machine ${MASTER_IP}"
  echo
  ssh $SSH_OPTS $MASTER "mkdir -p ~/kube/default"
  scp -r $SSH_OPTS saltbase/salt/generate-cert/make-ca-cert.sh ubuntu/reconfDocker.sh ubuntu/config-default.sh ubuntu/util.sh ubuntu/master/* ubuntu/binaries/master/ "${MASTER}:~/kube"

  # remote login to MASTER and use sudo to configue k8s master
  ssh $SSH_OPTS -t $MASTER "source ~/kube/util.sh; \
                            setClusterInfo; \
                            create-etcd-opts; \
                            create-kube-apiserver-opts "${SERVICE_CLUSTER_IP_RANGE}" "${ADMISSION_CONTROL}" "${SERVICE_NODE_PORT_RANGE}"; \
                            create-kube-controller-manager-opts "${NODE_IPS}"; \
                            create-kube-scheduler-opts; \
                            create-flanneld-opts "127.0.0.1"; \
                            sudo -p '[sudo] password to start master: ' cp ~/kube/default/* /etc/default/ && sudo cp ~/kube/init_conf/* /etc/init/ && sudo cp ~/kube/init_scripts/* /etc/init.d/ ;\
                            sudo groupadd -f -r kube-cert; \
                            ${PROXY_SETTING} sudo -E ~/kube/make-ca-cert.sh ${MASTER_IP} IP:${MASTER_IP},IP:${SERVICE_CLUSTER_IP_RANGE%.*}.1,DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.cluster.local; \
                            sudo mkdir -p /opt/bin/ && sudo cp ~/kube/master/* /opt/bin/; \
                            sudo service etcd start; \
                            sudo FLANNEL_NET=${FLANNEL_NET} -b ~/kube/reconfDocker.sh "a";"
}

function provision-node() {
  # copy the binaries and scripts to the ~/kube directory on the node
  echo "Deploying node on machine ${1#*@}"
  echo
  ssh $SSH_OPTS $1 "mkdir -p ~/kube/default"
  scp -r $SSH_OPTS ubuntu/config-default.sh ubuntu/util.sh ubuntu/reconfDocker.sh ubuntu/minion/* ubuntu/binaries/minion "${1}:~/kube"

  # remote login to MASTER and use sudo to configue k8s master
  ssh $SSH_OPTS -t $1 "source ~/kube/util.sh; \
                         setClusterInfo; \
                         create-kubelet-opts "${1#*@}" "${MASTER_IP}" "${DNS_SERVER_IP}" "${DNS_DOMAIN}"; \
                         create-kube-proxy-opts "${MASTER_IP}"; \
                         create-flanneld-opts "${MASTER_IP}"; \
                         sudo -p '[sudo] password to start node: ' cp ~/kube/default/* /etc/default/ && sudo cp ~/kube/init_conf/* /etc/init/ && sudo cp ~/kube/init_scripts/* /etc/init.d/ \
                         && sudo mkdir -p /opt/bin/ && sudo cp ~/kube/minion/* /opt/bin; \
                         sudo service flanneld start; \
                         sudo -b ~/kube/reconfDocker.sh "i";"
}

function provision-masterandnode() {
  # copy the binaries and scripts to the ~/kube directory on the master
  echo "Deploying master and node on machine ${MASTER_IP}"
  echo
  ssh $SSH_OPTS $MASTER "mkdir -p ~/kube/default"
  # scp order matters
  scp -r $SSH_OPTS saltbase/salt/generate-cert/make-ca-cert.sh ubuntu/config-default.sh ubuntu/util.sh ubuntu/minion/* ubuntu/master/* ubuntu/reconfDocker.sh ubuntu/binaries/master/ ubuntu/binaries/minion "${MASTER}:~/kube"

  # remote login to the node and use sudo to configue k8s
  ssh $SSH_OPTS -t $MASTER "source ~/kube/util.sh; \
                            setClusterInfo; \
                            create-etcd-opts; \
                            create-kube-apiserver-opts "${SERVICE_CLUSTER_IP_RANGE}" "${ADMISSION_CONTROL}" "${SERVICE_NODE_PORT_RANGE}"; \
                            create-kube-controller-manager-opts "${NODE_IPS}"; \
                            create-kube-scheduler-opts; \
                            create-kubelet-opts "${MASTER_IP}" "${MASTER_IP}" "${DNS_SERVER_IP}" "${DNS_DOMAIN}";
                            create-kube-proxy-opts "${MASTER_IP}";\
                            create-flanneld-opts "127.0.0.1"; \
                            sudo -p '[sudo] password to start master: ' cp ~/kube/default/* /etc/default/ && sudo cp ~/kube/init_conf/* /etc/init/ && sudo cp ~/kube/init_scripts/* /etc/init.d/ ; \
                            sudo groupadd -f -r kube-cert; \
                            ${PROXY_SETTING} sudo -E ~/kube/make-ca-cert.sh ${MASTER_IP} IP:${MASTER_IP},IP:${SERVICE_CLUSTER_IP_RANGE%.*}.1,DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.cluster.local; \
                            sudo mkdir -p /opt/bin/ && sudo cp ~/kube/master/* /opt/bin/ && sudo cp ~/kube/minion/* /opt/bin/; \
                            sudo service etcd start; \
                            sudo FLANNEL_NET=${FLANNEL_NET} -b ~/kube/reconfDocker.sh "ai";"
}

# Delete a kubernetes cluster
function kube-down {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"
  
  source "${KUBE_ROOT}/cluster/common.sh"
  tear_down_alive_resources

  ii=0
  for i in ${nodes}; do
    {
      echo "Cleaning on node ${i#*@}"
      if [[ "${roles[${ii}]}" == "ai" || "${roles[${ii}]}" == "a" ]]; then
        ssh -t $i 'pgrep etcd && sudo -p "[sudo] password to stop master: " service etcd stop && sudo rm -rf /infra*; 
          sudo rm -rf /opt/bin/etcd* /etc/init/etcd.conf /etc/init.d/etcd /etc/default/etcd'
      elif [[ "${roles[${ii}]}" == "i" ]]; then
        ssh -t $i 'pgrep flanneld && sudo -p "[sudo] password to stop node: " service flanneld stop'
      else
        echo "unsupported role for ${i}"
      fi
      # Delete the files in order to generate a clean environment, so you can change each node's role at next deployment.
      ssh -t $i 'sudo rm -f /opt/bin/kube* /opt/bin/flanneld;
      sudo rm -rf /etc/init/kube* /etc/init/flanneld.conf /etc/init.d/kube* /etc/init.d/flanneld;
      sudo rm -rf /etc/default/kube* /etc/default/flanneld; 
      sudo rm -rf ~/kube /var/lib/kubelet'
    }
    ((ii=ii+1))
  done
}


# Perform common upgrade setup tasks
function prepare-push() {
  # Use local binaries for kube-push
  if [[ "${KUBE_VERSION}" == "" ]]; then
    if [[ ! -d "${KUBE_ROOT}/cluster/ubuntu/binaries" ]]; then
      echo "No local binaries.Please check"
      exit 1
    else 
      echo "Please make sure all the required local binaries are prepared ahead"
      sleep 3
    fi
  else
    # Run build.sh to get the required release 
    export KUBE_VERSION
    "${KUBE_ROOT}/cluster/ubuntu/build.sh"
  fi
}

# Update a kubernetes master with required release
function push-master {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"

  if [[ ! -f "${KUBE_ROOT}/cluster/ubuntu/binaries/master/kube-apiserver" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi

  setClusterInfo
  ii=0
  for i in ${nodes}; do
    if [[ "${roles[${ii}]}" == "a" ]]; then
      echo "Cleaning master ${i#*@}"
      ssh -t $i 'sudo -p "[sudo] stop the all process: " service etcd stop;
      sudo rm -rf /opt/bin/etcd* /etc/init/etcd.conf /etc/init.d/etcd /etc/default/etcd;
      sudo rm -f /opt/bin/kube* /opt/bin/flanneld;
      sudo rm -rf /etc/init/kube* /etc/init/flanneld.conf /etc/init.d/kube* /etc/init.d/flanneld;
      sudo rm -rf /etc/default/kube* /etc/default/flanneld; 
      sudo rm -rf ~/kube' || true
      provision-master
    elif [[ "${roles[${ii}]}" == "ai" ]]; then 
      echo "Cleaning master ${i#*@}"
      ssh -t $i 'sudo -p "[sudo] stop the all process: " service etcd stop;
      sudo rm -rf /opt/bin/etcd* /etc/init/etcd.conf /etc/init.d/etcd /etc/default/etcd;
      sudo rm -f /opt/bin/kube* /opt/bin/flanneld;
      sudo rm -rf /etc/init/kube* /etc/init/flanneld.conf /etc/init.d/kube* /etc/init.d/flanneld;
      sudo rm -rf /etc/default/kube* /etc/default/flanneld; 
      sudo rm -rf ~/kube' || true
      provision-masterandnode
    elif [[ "${roles[${ii}]}" == "i" ]]; then
      ((ii=ii+1))
      continue
    else
      echo "unsupported role for ${i}, please check"
      exit 1
    fi
    ((ii=ii+1))
  done
  verify-cluster
}

# Update a kubernetes node with required release
function push-node() {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"

  if [[ ! -f "${KUBE_ROOT}/cluster/ubuntu/binaries/minion/kubelet" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi

  node_ip=${1}
  setClusterInfo
  ii=0
  existing=false
  for i in ${nodes}; do
    if [[ "${roles[${ii}]}" == "i" && ${i#*@} == $node_ip ]]; then
      echo "Cleaning node ${i#*@}"
      ssh -t $i 'sudo -p "[sudo] stop the all process: " service flanneld stop;
      sudo rm -f /opt/bin/kube* /opt/bin/flanneld;
      sudo rm -rf /etc/init/kube* /etc/init/flanneld.conf /etc/init.d/kube* /etc/init.d/flanneld;
      sudo rm -rf /etc/default/kube* /etc/default/flanneld; 
      sudo rm -rf ~/kube' || true
      provision-node $i
      existing=true
    elif [[ "${roles[${ii}]}" == "a" || "${roles[${ii}]}" == "ai" ]] && [[ ${i#*@} == $node_ip ]]; then
      echo "${i} is master node, please try ./kube-push -m instead"
      existing=true
    elif [[ "${roles[${ii}]}" == "i" || "${roles[${ii}]}" == "a" || "${roles[${ii}]}" == "ai" ]]; then
      ((ii=ii+1))
      continue
    else
      echo "unsupported role for ${i}, please check"
      exit 1
    fi
    ((ii=ii+1))
  done
  if [[ "${existing}" == false ]]; then
    echo "node ${node_ip} does not exist"
  else
    verify-cluster
  fi 
  
}

# Update a kubernetes cluster with required source
function kube-push { 
  prepare-push
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE-"config-default.sh"}"

  if [[ ! -f "${KUBE_ROOT}/cluster/ubuntu/binaries/master/kube-apiserver" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi

  #stop all the kube's process & etcd 
  ii=0
  for i in ${nodes}; do
    {
      echo "Cleaning on node ${i#*@}"
      if [[ "${roles[${ii}]}" == "ai" || "${roles[${ii}]}" == "a" ]]; then
        ssh -t $i 'pgrep etcd && sudo -p "[sudo] password to stop master: " service etcd stop; 
        sudo rm -rf /opt/bin/etcd* /etc/init/etcd.conf /etc/init.d/etcd /etc/default/etcd' || true
      elif [[ "${roles[${ii}]}" == "i" ]]; then
        ssh -t $i 'pgrep flanneld && sudo -p "[sudo] password to stop node: " service flanneld stop' || true
      else
        echo "unsupported role for ${i}"
      fi

      ssh -t $i 'sudo rm -f /opt/bin/kube* /opt/bin/flanneld;
      sudo rm -rf /etc/init/kube* /etc/init/flanneld.conf /etc/init.d/kube* /etc/init.d/flanneld;
      sudo rm -rf /etc/default/kube* /etc/default/flanneld; 
      sudo rm -rf ~/kube' || true
    }
    ((ii=ii+1))
  done

  #provision all nodes,including master & nodes
  setClusterInfo
  ii=0
  for i in ${nodes}; do
    if [[ "${roles[${ii}]}" == "a" ]]; then
      provision-master
    elif [[ "${roles[${ii}]}" == "i" ]]; then
      provision-node $i
    elif [[ "${roles[${ii}]}" == "ai" ]]; then
      provision-masterandnode
    else
      echo "unsupported role for ${i}. please check"
      exit 1
    fi
    ((ii=ii+1))
  done
  verify-cluster
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  echo "Ubuntu doesn't need special preparations for e2e tests" 1>&2
}
