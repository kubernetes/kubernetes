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

# A library of helper functions that each provider hosting Kubernetes
# must implement to use cluster/kube-*.sh scripts.
set -e

SSH_OPTS="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR"

MASTER=""
MASTER_IP=""
NODE_IPS=""

# Assumed Vars:
#   KUBE_ROOT
function test-build-release() {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# From user input set the necessary k8s and etcd configuration information
function setClusterInfo() {
  # Initialize NODE_IPS in setClusterInfo function
  # NODE_IPS is defined as a global variable, and is concatenated with other nodeIP	
  # When setClusterInfo is called for many times, this could cause potential problems
  # Such as, you will have NODE_IPS=192.168.0.2,192.168.0.3,192.168.0.2,192.168.0.3,
  # which is obviously wrong.
  NODE_IPS=""
  
  local ii=0
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
function verify-prereqs() {
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
function trap-add() {
  local handler="$1"
  local signal="${2-EXIT}"
  local cur

  cur="$(eval "sh -c 'echo \$3' -- $(trap -p ${signal})")"
  if [[ -n "${cur}" ]]; then
    handler="${cur}; ${handler}"
  fi

  trap "${handler}" ${signal}
}

function verify-cluster() {
  local ii=0

  for i in ${nodes}
  do
    if [ "${roles[${ii}]}" == "a" ]; then
      verify-master
    elif [ "${roles[${ii}]}" == "i" ]; then
      verify-node "$i"
    elif [ "${roles[${ii}]}" == "ai" ]; then
      verify-master
      verify-node "$i"
    else
      echo "unsupported role for ${i}. please check"
      exit 1
    fi

    ((ii=ii+1))
  done

}

function verify-master() {
  # verify master has all required daemons
  echo -n "Validating master"
  local -a required_daemon=("kube-apiserver" "kube-controller-manager" "kube-scheduler")
  local validated="1"
  local try_count=1
  local max_try_count=30
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
      ssh $SSH_OPTS "$MASTER" "pgrep -f '${daemon}'" >/dev/null 2>&1 || {
        echo -n "."
        validated="1"
        ((try_count=try_count+1))
        if [[ ${try_count} -gt ${max_try_count} ]]; then
          echo -e "\nWarning: Process '${daemon}' failed to run on ${MASTER}, please check.\n"
          exit 1
        fi
        sleep 2
      }
    done
  done
  echo

}

function verify-node() {
  # verify node has all required daemons
  echo -n "Validating ${1}"
  local -a required_daemon=("kube-proxy" "kubelet" "docker")
  local validated="1"
  local try_count=1
  local max_try_count=30
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
      ssh $SSH_OPTS "$1" "pgrep -f '${daemon}'" >/dev/null 2>&1 || {
        echo -n "."
        validated="1"
        ((try_count=try_count+1))
        if [[ ${try_count} -gt ${max_try_count} ]]; then
          echo -e "\nWarning: Process '${daemon}' failed to run on ${1}, please check.\n"
          exit 1
        fi
        sleep 2
      }
    done
  done
  echo
}

function create-etcd-opts() {
  cat <<EOF > ~/kube/default/etcd
ETCD_OPTS="\
 -name infra\
 -listen-client-urls http://127.0.0.1:4001,http://${1}:4001\
 -advertise-client-urls http://${1}:4001"
EOF
}

function create-kube-apiserver-opts() {
  cat <<EOF > ~/kube/default/kube-apiserver
KUBE_APISERVER_OPTS="\
 --insecure-bind-address=0.0.0.0\
 --insecure-port=8080\
 --etcd-servers=http://127.0.0.1:4001\
 --logtostderr=true\
 --service-cluster-ip-range=${1}\
 --admission-control=${2}\
 --service-node-port-range=${3}\
 --client-ca-file=/srv/kubernetes/ca.crt\
 --tls-cert-file=/srv/kubernetes/server.cert\
 --tls-private-key-file=/srv/kubernetes/server.key"
EOF
}

function create-kube-controller-manager-opts() {
  cat <<EOF > ~/kube/default/kube-controller-manager
KUBE_CONTROLLER_MANAGER_OPTS="\
 --master=127.0.0.1:8080\
 --root-ca-file=/srv/kubernetes/ca.crt\
 --service-account-private-key-file=/srv/kubernetes/server.key\
 --logtostderr=true"
EOF

}

function create-kube-scheduler-opts() {
  cat <<EOF > ~/kube/default/kube-scheduler
KUBE_SCHEDULER_OPTS="\
 --logtostderr=true\
 --master=127.0.0.1:8080"
EOF

}

function create-kubelet-opts() {
  cat <<EOF > ~/kube/default/kubelet
KUBELET_OPTS="\
 --address=0.0.0.0\
 --port=10250 \
 --hostname-override=${1} \
 --api-servers=http://${2}:8080 \
 --logtostderr=true \
 --cluster-dns=$3 \
 --cluster-domain=$4"
EOF

}

function create-kube-proxy-opts() {
  cat <<EOF > ~/kube/default/kube-proxy
KUBE_PROXY_OPTS="\
 --master=http://${1}:8080 \
 --logtostderr=true"
EOF

}

function create-flanneld-opts() {
  cat <<EOF > ~/kube/default/flanneld
FLANNEL_OPTS="--etcd-endpoints=http://${1}:4001"
EOF
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
# Vars set:
#   KUBE_MASTER_IP
function detect-master() {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE:-config-default.sh}"
  setClusterInfo
  export KUBE_MASTER="${MASTER}" 
  export KUBE_MASTER_IP="${MASTER_IP}"
  echo "Using master ${MASTER_IP}"
}

# Detect the information about the nodes
#
# Assumed vars:
#   nodes
# Vars set:
#   KUBE_NODE_IP_ADDRESS (array)
function detect-nodes() {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE:-config-default.sh}"

  KUBE_NODE_IP_ADDRESSES=()
  setClusterInfo

  local ii=0
  for i in ${nodes}
  do
    if [ "${roles[${ii}]}" == "i" ] || [ "${roles[${ii}]}" == "ai" ]; then
      KUBE_NODE_IP_ADDRESSES+=("${i#*@}")
    fi

    ((ii=ii+1))
  done

  if [[ -z "${KUBE_NODE_IP_ADDRESSES[@]}" ]]; then
    echo "Could not detect Kubernetes node nodes.\
    Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Instantiate a kubernetes cluster on ubuntu
function kube-up() {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE:-config-default.sh}"

  # downloading tarball release
  if [[ -d "${KUBE_ROOT}/cluster/ubuntu/binaries" ]]; then
    rm -rf "${KUBE_ROOT}/cluster/ubuntu/binaries"
  fi
  "${KUBE_ROOT}/cluster/ubuntu/download-release.sh"

  setClusterInfo
  local ii=0

  for i in ${nodes}
  do
    {
      if [ "${roles[${ii}]}" == "a" ]; then
        provision-master
      elif [ "${roles[${ii}]}" == "ai" ]; then
        provision-masterandnode
      elif [ "${roles[${ii}]}" == "i" ]; then
        provision-node "$i"
      else
        echo "unsupported role for ${i}. Please check"
        exit 1
      fi
    }

    ((ii=ii+1))
  done
  wait

  export KUBECTL_PATH="${KUBE_ROOT}/cluster/ubuntu/binaries/kubectl"
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
  
  echo -e "\nDeploying master on machine ${MASTER_IP}"

  ssh $SSH_OPTS "$MASTER" "mkdir -p ~/kube/default"

  # copy the binaries and scripts to the ~/kube directory on the master
  scp -r $SSH_OPTS \
    saltbase/salt/generate-cert/make-ca-cert.sh \
    ubuntu/reconfDocker.sh \
    ubuntu/${KUBE_CONFIG_FILE:-config-default.sh} \
    ubuntu/util.sh \
    ubuntu/master/* \
    ubuntu/binaries/master/ \
    "${MASTER}:~/kube"

  EXTRA_SANS=(
    IP:$MASTER_IP
    IP:${SERVICE_CLUSTER_IP_RANGE%.*}.1
    DNS:kubernetes
    DNS:kubernetes.default
    DNS:kubernetes.default.svc
    DNS:kubernetes.default.svc.cluster.local
  )
  
  EXTRA_SANS=$(echo "${EXTRA_SANS[@]}" | tr ' ' ,)

  # remote login to MASTER and configue k8s master
  ssh $SSH_OPTS -t "${MASTER}" "
    source ~/kube/util.sh

    setClusterInfo
    create-etcd-opts '${MASTER_IP}'
    create-kube-apiserver-opts \
      '${SERVICE_CLUSTER_IP_RANGE}' \
      '${ADMISSION_CONTROL}' \
      '${SERVICE_NODE_PORT_RANGE}'
    create-kube-controller-manager-opts '${NODE_IPS}'
    create-kube-scheduler-opts
    create-flanneld-opts '127.0.0.1'
    sudo -E -p '[sudo] password to start master: ' -- /bin/bash -c '
      cp ~/kube/default/* /etc/default/ 
      cp ~/kube/init_conf/* /etc/init/ 
      cp ~/kube/init_scripts/* /etc/init.d/
      
      groupadd -f -r kube-cert
      \"${PROXY_SETTING}\" ~/kube/make-ca-cert.sh \"${MASTER_IP}\" \"${EXTRA_SANS}\"
      mkdir -p /opt/bin/
      cp ~/kube/master/* /opt/bin/
      service etcd start
      FLANNEL_NET=\"${FLANNEL_NET}\" ~/kube/reconfDocker.sh a
      '" || {
      echo "Deploying master on machine ${MASTER_IP} failed"
      exit 1
    }
} 

function provision-node() {
  
  echo -e "\nDeploying node on machine ${1#*@}"

  ssh $SSH_OPTS $1 "mkdir -p ~/kube/default"

  # copy the binaries and scripts to the ~/kube directory on the node
  scp -r $SSH_OPTS \
    ubuntu/${KUBE_CONFIG_FILE:-config-default.sh} \
    ubuntu/util.sh \
    ubuntu/reconfDocker.sh \
    ubuntu/minion/* \
    ubuntu/binaries/minion \
    "${1}:~/kube"

  # remote login to node and configue k8s node
  ssh $SSH_OPTS -t "$1" "
    source ~/kube/util.sh
    
    setClusterInfo
    create-kubelet-opts \
      '${1#*@}' \
      '${MASTER_IP}' \
      '${DNS_SERVER_IP}' \
      '${DNS_DOMAIN}'
    create-kube-proxy-opts '${MASTER_IP}'
    create-flanneld-opts '${MASTER_IP}'
                         
    sudo -E -p '[sudo] password to start node: ' -- /bin/bash -c '
      cp ~/kube/default/* /etc/default/
      cp ~/kube/init_conf/* /etc/init/
      cp ~/kube/init_scripts/* /etc/init.d/ 
      mkdir -p /opt/bin/ 
      cp ~/kube/minion/* /opt/bin
      service flanneld start
      ~/kube/reconfDocker.sh i
      '" || {
      echo "Deploying node on machine ${1#*@} failed"
      exit 1
  }
}

function provision-masterandnode() {
  
  echo -e "\nDeploying master and node on machine ${MASTER_IP}"

  ssh $SSH_OPTS $MASTER "mkdir -p ~/kube/default"

  # copy the binaries and scripts to the ~/kube directory on the master
  # scp order matters
  scp -r $SSH_OPTS \
    saltbase/salt/generate-cert/make-ca-cert.sh \
    ubuntu/${KUBE_CONFIG_FILE:-config-default.sh} \
    ubuntu/util.sh \
    ubuntu/minion/* \
    ubuntu/master/* \
    ubuntu/reconfDocker.sh \
    ubuntu/binaries/master/ \
    ubuntu/binaries/minion \
    "${MASTER}:~/kube"
  
  EXTRA_SANS=(
    IP:${MASTER_IP}
    IP:${SERVICE_CLUSTER_IP_RANGE%.*}.1
    DNS:kubernetes
    DNS:kubernetes.default
    DNS:kubernetes.default.svc
    DNS:kubernetes.default.svc.cluster.local
  )
  
  EXTRA_SANS=$(echo "${EXTRA_SANS[@]}" | tr ' ' ,)

  # remote login to the master/node and configue k8s
  ssh $SSH_OPTS -t "$MASTER" "
    source ~/kube/util.sh
     
    setClusterInfo
    create-etcd-opts '${MASTER_IP}'
    create-kube-apiserver-opts \
      '${SERVICE_CLUSTER_IP_RANGE}' \
      '${ADMISSION_CONTROL}' \
      '${SERVICE_NODE_PORT_RANGE}'
    create-kube-controller-manager-opts '${NODE_IPS}'
    create-kube-scheduler-opts
    create-kubelet-opts \
      '${MASTER_IP}' \
      '${MASTER_IP}' \
      '${DNS_SERVER_IP}' \
      '${DNS_DOMAIN}'
    create-kube-proxy-opts '${MASTER_IP}'
    create-flanneld-opts '127.0.0.1'
    
    sudo -E -p '[sudo] password to start master: ' -- /bin/bash -c ' 
      cp ~/kube/default/* /etc/default/ 
      cp ~/kube/init_conf/* /etc/init/ 
      cp ~/kube/init_scripts/* /etc/init.d/
      
      groupadd -f -r kube-cert
      \"${PROXY_SETTING}\" ~/kube/make-ca-cert.sh \"${MASTER_IP}\" \"${EXTRA_SANS}\"
      mkdir -p /opt/bin/ 
      cp ~/kube/master/* /opt/bin/
      cp ~/kube/minion/* /opt/bin/

      service etcd start
      FLANNEL_NET=\"${FLANNEL_NET}\" ~/kube/reconfDocker.sh ai
      '" || {
      echo "Deploying master and node on machine ${MASTER_IP} failed"
      exit 1
  }    
}

# check whether kubelet has torn down all of the pods
function check-pods-torn-down() {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local attempt=0
  while [[ ! -z "$(kubectl get pods | tail -n +2)" ]]; do
    if (( attempt > 120 )); then
      echo "timeout waiting for tearing down pods" >> ~/kube/err.log
    fi
    echo "waiting for tearing down pods"
    attempt=$((attempt+1))
    sleep 5
  done
}

# Delete a kubernetes cluster
function kube-down() {
  
  export KUBECTL_PATH="${KUBE_ROOT}/cluster/ubuntu/binaries/kubectl"
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE:-config-default.sh}"
  source "${KUBE_ROOT}/cluster/common.sh"

  tear_down_alive_resources
  check-pods-torn-down 
  
  local ii=0
  for i in ${nodes}; do
      if [[ "${roles[${ii}]}" == "ai" || "${roles[${ii}]}" == "a" ]]; then
        echo "Cleaning on master ${i#*@}"
        ssh $SSH_OPTS -t "$i" "
          pgrep etcd && \
          sudo -p '[sudo] password to stop master: ' -- /bin/bash -c '
            service etcd stop

            rm -rf \
              /opt/bin/etcd* \
              /etc/init/etcd.conf \
              /etc/init.d/etcd \
              /etc/default/etcd
      
            rm -rf /infra*
            rm -rf /srv/kubernetes
            '
        " || echo "Cleaning on master ${i#*@} failed"

        if [[ "${roles[${ii}]}" == "ai" ]]; then
          ssh $SSH_OPTS -t "$i" "sudo rm -rf /var/lib/kubelet"
        fi
        
      elif [[ "${roles[${ii}]}" == "i" ]]; then
        echo "Cleaning on node ${i#*@}"
        ssh $SSH_OPTS -t "$i" "
          pgrep flanneld && \
          sudo -p '[sudo] password to stop node: ' -- /bin/bash -c '
            service flanneld stop
            rm -rf /var/lib/kubelet            
            '
          " || echo "Cleaning on node ${i#*@} failed"
      else
        echo "unsupported role for ${i}"
      fi
      
      ssh $SSH_OPTS -t "$i" "sudo -- /bin/bash -c '
        rm -f \
          /opt/bin/kube* \
          /opt/bin/flanneld \
          /etc/init/kube* \
          /etc/init/flanneld.conf \
          /etc/init.d/kube* \
          /etc/init.d/flanneld \
          /etc/default/kube* \
          /etc/default/flanneld
        
        rm -rf ~/kube
        rm -f /run/flannel/subnet.env
      '" || echo "cleaning legacy files on ${i#*@} failed"
    ((ii=ii+1))
  done
}


# Perform common upgrade setup tasks
function prepare-push() {
  # Use local binaries for kube-push
  if [[ -z "${KUBE_VERSION}" ]]; then
    echo "Use local binaries for kube-push" 
    if [[ ! -d "${KUBE_ROOT}/cluster/ubuntu/binaries" ]]; then
      echo "No local binaries.Please check"
      exit 1
    else 
      echo "Please make sure all the required local binaries are prepared ahead"
      sleep 3
    fi
  else
    # Run download-release.sh to get the required release 
    export KUBE_VERSION
    "${KUBE_ROOT}/cluster/ubuntu/download-release.sh"
  fi
}

# Update a kubernetes master with expected release
function push-master() {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE:-config-default.sh}"
  
  if [[ ! -f "${KUBE_ROOT}/cluster/ubuntu/binaries/master/kube-apiserver" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi
  export KUBECTL_PATH="${KUBE_ROOT}/cluster/ubuntu/binaries/kubectl"
  
  setClusterInfo

  local ii=0
  for i in ${nodes}; do
    if [[ "${roles[${ii}]}" == "a" || "${roles[${ii}]}" == "ai" ]]; then
      echo "Cleaning master ${i#*@}"
      ssh $SSH_OPTS -t "$i" "
        pgrep etcd && sudo -p '[sudo] stop the all process: ' -- /bin/bash -c '
        service etcd stop
        sleep 3
        rm -rf \
          /etc/init/etcd.conf \
          /etc/init/kube* \
          /etc/init/flanneld.conf \
          /etc/init.d/etcd \
          /etc/init.d/kube* \
          /etc/init.d/flanneld \
          /etc/default/etcd \
          /etc/default/kube* \
          /etc/default/flanneld
        rm -f \
          /opt/bin/etcd* \
          /opt/bin/kube* \
          /opt/bin/flanneld
        rm -f /run/flannel/subnet.env
        rm -rf ~/kube
      '" || echo "Cleaning master ${i#*@} failed"
    fi 
    
    if [[ "${roles[${ii}]}" == "a" ]]; then
      provision-master
    elif [[ "${roles[${ii}]}" == "ai" ]]; then
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

# Update a kubernetes node with expected release
function push-node() {
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE:-config-default.sh}"

  if [[ ! -f "${KUBE_ROOT}/cluster/ubuntu/binaries/minion/kubelet" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi

  export KUBECTL_PATH="${KUBE_ROOT}/cluster/ubuntu/binaries/kubectl"
  
  setClusterInfo
  
  local node_ip=${1}
  local ii=0
  local existing=false

  for i in ${nodes}; do
    if [[ "${roles[${ii}]}" == "i" && ${i#*@} == "$node_ip" ]]; then
      echo "Cleaning node ${i#*@}"
      ssh $SSH_OPTS -t "$i" "
        sudo -p '[sudo] stop the all process: ' -- /bin/bash -c '
          service flanneld stop

          rm -f /opt/bin/kube* \
            /opt/bin/flanneld

          rm -rf \
            /etc/init/kube* \
            /etc/init/flanneld.conf \
            /etc/init.d/kube* \
            /etc/init.d/flanneld \
            /etc/default/kube* \
            /etc/default/flanneld

          rm -f /run/flannel/subnet.env

          rm -rf ~/kube
        '" || echo "Cleaning node ${i#*@} failed"
      provision-node "$i"
      existing=true
    elif [[ "${roles[${ii}]}" == "a" || "${roles[${ii}]}" == "ai" ]] && [[ ${i#*@} == "$node_ip" ]]; then
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

# Update a kubernetes cluster with expected source
function kube-push() { 
  prepare-push
  source "${KUBE_ROOT}/cluster/ubuntu/${KUBE_CONFIG_FILE:-config-default.sh}"

  if [[ ! -f "${KUBE_ROOT}/cluster/ubuntu/binaries/master/kube-apiserver" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi
  
  export KUBECTL_PATH="${KUBE_ROOT}/cluster/ubuntu/binaries/kubectl"
  #stop all the kube's process & etcd 
  local ii=0
  for i in ${nodes}; do
     if [[ "${roles[${ii}]}" == "ai" || "${roles[${ii}]}" == "a" ]]; then
       echo "Cleaning on master ${i#*@}"
       ssh $SSH_OPTS -t "$i" "
        pgrep etcd && \
        sudo -p '[sudo] password to stop master: ' -- /bin/bash -c '
          service etcd stop

          rm -rf \
            /opt/bin/etcd* \
            /etc/init/etcd.conf \
            /etc/init.d/etcd \
            /etc/default/etcd
        '" || echo "Cleaning on master ${i#*@} failed"
      elif [[ "${roles[${ii}]}" == "i" ]]; then
        echo "Cleaning on node ${i#*@}"
        ssh $SSH_OPTS -t $i "
        pgrep flanneld && \
        sudo -p '[sudo] password to stop node: ' -- /bin/bash -c '
          service flanneld stop
        '" || echo "Cleaning on node ${i#*@} failed"
      else
        echo "unsupported role for ${i}"
      fi

      ssh $SSH_OPTS -t "$i" "sudo -- /bin/bash -c '
        rm -f \
          /opt/bin/kube* \
          /opt/bin/flanneld

        rm -rf \
          /etc/init/kube* \
          /etc/init/flanneld.conf \
          /etc/init.d/kube* \
          /etc/init.d/flanneld \
          /etc/default/kube* \
          /etc/default/flanneld

        rm -f /run/flannel/subnet.env
        rm -rf ~/kube
      '" || echo "Cleaning legacy files on ${i#*@} failed"
    ((ii=ii+1))
  done

  #provision all nodes,including master & nodes
  setClusterInfo

  local ii=0
  for i in ${nodes}; do
    if [[ "${roles[${ii}]}" == "a" ]]; then
      provision-master
    elif [[ "${roles[${ii}]}" == "i" ]]; then
      provision-node "$i"
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
