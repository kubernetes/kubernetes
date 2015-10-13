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

# A library of helper functions that each provider hosting Kubernetes must
# implement to use cluster/kube-*.sh scripts.
set -e

MASTER=
MASTER_IP=
NODE_IPS=


BASE_DIR="$KUBE_ROOT/cluster/ubuntu"

SSH_OPTS="\
 -oStrictHostKeyChecking=no\
 -oUserKnownHostsFile=/dev/null\
 -oLogLevel=ERROR\
"

# Assumed Vars:
#   KUBE_ROOT
function test-build-release {
  # Make a release
  "$KUBE_ROOT/build/release.sh"
}

# From user input set the necessary k8s and etcd configuration information
function setClusterInfo() {
  # Initialize NODE_IPS in setClusterInfo function
  #
  # NOTE:
  # NODE_IPS is defined as a global variable, and it's concatenated with other
  # nodeIP. Thus, if setClusterInfo is called many times, it could cause
  # potential problems, because you will have multiple IPs in NODE_IPS which is
  # most probably wrong.
  NODE_IPS=""

  local i=0
  for node in $nodes; do
    nodeIP=${node#*@}

    if [[ "${roles[${i}]}" == "ai" ]]; then
      MASTER="$node"
      MASTER_IP="$nodeIP"
      NODE_IPS="$nodeIP"
    elif [[ "${roles[${i}]}" == "a" ]]; then
      MASTER="$node"
      MASTER_IP="$nodeIP"
    elif [[ "${roles[${i}]}" == "i" ]]; then
      if [[ -z $NODE_IPS ]];then
        NODE_IPS="$nodeIP"
      else
        NODE_IPS="$NODE_IPS,$nodeIP"
      fi
    else
      echo "ERROR: unsupported role for $node. Please check"
      exit 1
    fi

    ((i=i+1))
  done
}


# Verify ssh prereqs
function verify-prereqs {
  local rc

  rc=0
  ssh-add -L 1> /dev/null 2> /dev/null || rc="$?"
  # "Could not open a connection to your authentication agent."
  if [[ "$rc" -eq 2 ]]; then
    eval "$(ssh-agent)" > /dev/null
    trap-add "kill $SSH_AGENT_PID" EXIT
  fi

  rc=0
  ssh-add -L 1> /dev/null 2> /dev/null || rc="$?"
  # "The agent has no identities."
  if [[ "$rc" -eq 1 ]]; then
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

  cur="$(eval "sh -c 'echo \$3' -- $(trap -p $signal)")"
  if [[ -n "$cur" ]]; then
    handler="$cur; $handler"
  fi

  trap "$handler" $signal
}

function verify-cluster {
  local i=0
  for node in $nodes; do
    if [ "${roles[${i}]}" == "a" ]; then
      verify-master
    elif [ "${roles[${i}]}" == "i" ]; then
      verify-node $node
    elif [ "${roles[${i}]}" == "ai" ]; then
      verify-master
      verify-node $node
    else
      echo "ERROR: unsupported role for $node. Please check"
      exit 1
    fi

    ((i=i+1))
  done

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  http://$MASTER_IP:8080"
  echo
}

function verify-master {
  verify-server \
      "master" \
      "$MASTER" \
      "kube-apiserver kube-controller-manager kube-scheduler"
}

function verify-node {
  verify-server \
      "node" \
      "$1" \
      "kube-proxy kubelet docker"
}

function verify-server {
  local ttype="$1"
  local server="$2"
  local required_daemon="$3"

  # verify node has all required daemons
  echo -n "Validating $ttype [$server]"
  local validated="1"
  local try_count=1
  local max_try_count=30
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in $required_daemon; do
      ssh $SSH_OPTS "$server" "pgrep -f $daemon" >/dev/null 2>&1 || {
        echo -n "."
        validated="1"
        ((try_count=try_count+1))
        if (($try_count > $max_try_count)); then
          echo -en "\nWARNING: process [$daemon] failed to run on $ttype "
          echo -e "[$server], please check.\n"
          exit 1
        fi
        sleep 2
      }
    done
  done

  echo
}

function create-etcd-opts {
  cat <<EOF > ~/kube/default/etcd
ETCD_OPTS="\
 -name infra\
 -listen-client-urls http://0.0.0.0:4001\
 -advertise-client-urls http://127.0.0.1:4001\
"
EOF
}

function create-kube-apiserver-opts {
  cat <<EOF > ~/kube/default/kube-apiserver
KUBE_APISERVER_OPTS="\
 --insecure-bind-address=0.0.0.0\
 --insecure-port=8080\
 --etcd-servers=http://127.0.0.1:4001\
 --logtostderr=true\
 --service-cluster-ip-range=$1\
 --admission-control=$2\
 --service-node-port-range=$3\
 --client-ca-file=/srv/kubernetes/ca.crt\
 --tls-cert-file=/srv/kubernetes/server.cert\
 --tls-private-key-file=/srv/kubernetes/server.key\
"
EOF
}

function create-kube-controller-manager-opts {
  cat <<EOF > ~/kube/default/kube-controller-manager
KUBE_CONTROLLER_MANAGER_OPTS="\
 --master=127.0.0.1:8080\
 --root-ca-file=/srv/kubernetes/ca.crt\
 --service-account-private-key-file=/srv/kubernetes/server.key\
 --logtostderr=true\
"
EOF
}

function create-kube-scheduler-opts {
  cat <<EOF > ~/kube/default/kube-scheduler
KUBE_SCHEDULER_OPTS="
 --logtostderr=true\
 --master=127.0.0.1:8080\
"
EOF
}

function create-kubelet-opts {
  cat <<EOF > ~/kube/default/kubelet
KUBELET_OPTS="\
 --address=0.0.0.0\
 --port=10250\
 --hostname-override=$1\
 --api-servers=http://$2:8080\
 --logtostderr=true\
 --cluster-dns=$3\
 --cluster-domain=$4\
"
EOF
}

function create-kube-proxy-opts {
  cat <<EOF > ~/kube/default/kube-proxy
KUBE_PROXY_OPTS="\
  --master=http://$1:8080 \
  --logtostderr=true"
EOF

}

function create-flanneld-opts {
  cat <<EOF > ~/kube/default/flanneld
FLANNEL_OPTS="\
 --etcd-endpoints=http://$1:4001\
"
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
  source "$BASE_DIR/${KUBE_CONFIG_FILE-"config-default.sh"}"
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
  source "$BASE_DIR/${KUBE_CONFIG_FILE-"config-default.sh"}"

  KUBE_NODE_IP_ADDRESSES=()

  setClusterInfo

  local i=0
  for node in $nodes; do
    if [[ "${roles[${i}]}" == "i" || "${roles[${i}]}" == "ai" ]]; then
      KUBE_NODE_IP_ADDRESSES+=("${node#*@}")
    fi

    ((i=i+1))
  done

  if [[ -z ${KUBE_NODE_IP_ADDRESSES[@]} ]]; then
    echo -n "Could not detect Kubernetes node nodes. " >&2
    echo "Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Instantiate a kubernetes cluster on ubuntu
function kube-up {
  source "$BASE_DIR/${KUBE_CONFIG_FILE-"config-default.sh"}"

  # ensure the binaries are well prepared
  if [ ! -f "ubuntu/binaries/master/kube-apiserver" ]; then
    echo "No local binaries for kube-up, downloading..."
    $BASE_DIR/build.sh
  fi

  setClusterInfo

  local i=0
  for node in $nodes; do
    if [ "${roles[${i}]}" == "a" ]; then
      provision-master
    elif [ "${roles[${i}]}" == "ai" ]; then
      provision-masterandnode
    elif [ "${roles[${i}]}" == "i" ]; then
      provision-node $node
    else
      echo "ERROR: unsupported role for $node. Please check"
      exit 1
    fi

    ((i=i+1))
  done
  wait

  verify-cluster
  detect-master
  export CONTEXT="ubuntu"
  export KUBE_SERVER="http://$KUBE_MASTER_IP:8080"

  source "$KUBE_ROOT/cluster/common.sh"

  # set kubernetes user and password
  gen-kube-basicauth

  create-kubeconfig
}

function provision-master {
  echo "provision-master: deploying master on machine $MASTER_IP"
  echo
  ssh $SSH_OPTS $MASTER "mkdir -p ~/kube/default"
  scp -r $SSH_OPTS \
    saltbase/salt/generate-cert/make-ca-cert.sh \
    ubuntu/reconfDocker.sh \
    ubuntu/config-default.sh \
    ubuntu/util.sh \
    ubuntu/master/* \
    ubuntu/binaries/master/ \
    "$MASTER:~/kube"

  EXTRA_SANS=(
    IP:$MASTER_IP
    IP:${SERVICE_CLUSTER_IP_RANGE%.*}.1
    DNS:kubernetes
    DNS:kubernetes.default
    DNS:kubernetes.default.svc
    DNS:kubernetes.default.svc.cluster.local
  )
  EXTRA_SANS=$(echo ${EXTRA_SANS[@]} | tr ' ' ',')

  ssh $SSH_OPTS -t $MASTER '
    source ~/kube/util.sh

    setClusterInfo
    create-etcd-opts
    create-kube-apiserver-opts \
      "'$SERVICE_CLUSTER_IP_RANGE'" \
      "'$ADMISSION_CONTROL'" \
      "'$SERVICE_NODE_PORT_RANGE'"
    create-kube-controller-manager-opts "'$NODE_IPS'"
    create-kube-scheduler-opts
    create-flanneld-opts "127.0.0.1"

    sudo -E -p "[sudo] password to start master: " -- /bin/bash -c "
      cp ~/kube/default/* /etc/default/
      cp ~/kube/init_conf/* /etc/init/
      cp ~/kube/init_scripts/* /etc/init.d/

      groupadd -f -r kube-cert
      '$PROXY_SETTING' ~/kube/make-ca-cert.sh "'$MASTER_IP'" "'$EXTRA_SANS'"

      mkdir -p /opt/bin/ && cp ~/kube/master/* /opt/bin/

      service etcd start
      FLANNEL_NET="'$FLANNEL_NET'" ~/kube/reconfDocker.sh a
    "
  ' || {
    echo "provision-master: ssh failed"
    exit 1
  }
}

function provision-node {
  echo "provision-node: deploying node on machine ${1#*@}"
  echo

  ssh $SSH_OPTS $1 "mkdir -p ~/kube/default"

  scp -r $SSH_OPTS \
    ubuntu/config-default.sh \
    ubuntu/util.sh \
    ubuntu/reconfDocker.sh \
    ubuntu/minion/* \
    ubuntu/binaries/minion \
    "$1:~/kube"

  ssh $SSH_OPTS -t $1 '
    source ~/kube/util.sh

    setClusterInfo
    create-kubelet-opts \
      "'${1#*@}'" \
      "'$MASTER_IP'" \
      "'$DNS_SERVER_IP'" \
      "'$DNS_DOMAIN'"
    create-kube-proxy-opts "'$MASTER_IP'"
    create-flanneld-opts "'$MASTER_IP'"

    sudo -E -p "[sudo] password to start node: " -- /bin/bash -c "
      cp ~/kube/default/* /etc/default/
      cp ~/kube/init_conf/* /etc/init/
      cp ~/kube/init_scripts/* /etc/init.d/

      mkdir -p /opt/bin/ && cp ~/kube/minion/* /opt/bin

      service flanneld start
      ~/kube/reconfDocker.sh i
    "
  ' || {
    echo "provision-node: ssh failed"
    exit 1
  }
}

function provision-masterandnode {
  echo "provision-masterandnode: deploying master and node on machine ${MASTER_IP}"
  echo

  ssh $SSH_OPTS $MASTER "mkdir -p ~/kube/default"

  scp -r $SSH_OPTS \
    saltbase/salt/generate-cert/make-ca-cert.sh \
    ubuntu/config-default.sh \
    ubuntu/util.sh \
    ubuntu/minion/* \
    ubuntu/master/* \
    ubuntu/reconfDocker.sh \
    ubuntu/binaries/master/ \
    ubuntu/binaries/minion \
    "$MASTER:~/kube"

  EXTRA_SANS=(
    IP:${MASTER_IP}
    IP:${SERVICE_CLUSTER_IP_RANGE%.*}.1
    DNS:kubernetes
    DNS:kubernetes.default
    DNS:kubernetes.default.svc
    DNS:kubernetes.default.svc.cluster.local
  )
  EXTRA_SANS=$(echo ${EXTRA_SANS[@]} | tr ' ' ',')

  ssh $SSH_OPTS -t $MASTER '
    source ~/kube/util.sh

    setClusterInfo
    create-etcd-opts
    create-kube-apiserver-opts \
      "'$SERVICE_CLUSTER_IP_RANGE'" \
      "'$ADMISSION_CONTROL'" \
      "'$SERVICE_NODE_PORT_RANGE'"
    create-kube-controller-manager-opts "'$NODE_IPS'"
    create-kube-scheduler-opts
    create-kubelet-opts \
      "'$MASTER_IP'" \
      "'$MASTER_IP'" \
      "'$DNS_SERVER_IP'" \
      "'$DNS_DOMAIN'"
    create-kube-proxy-opts "'$MASTER_IP'"
    create-flanneld-opts "127.0.0.1"

    sudo -E -p "[sudo] password to start master: " -- /bin/bash -c "
      cp ~/kube/default/* /etc/default/
      cp ~/kube/init_conf/* /etc/init/
      cp ~/kube/init_scripts/* /etc/init.d/

      groupadd -f -r kube-cert
      '$PROXY_SETTING' ~/kube/make-ca-cert.sh '$MASTER_IP' '$EXTRA_SANS'

      mkdir -p /opt/bin/ && \
        cp ~/kube/master/* /opt/bin/ && \
        cp ~/kube/minion/* /opt/bin/

      service etcd start
      FLANNEL_NET='$FLANNEL_NET' ~/kube/reconfDocker.sh ai
    "
  ' || {
    echo "provision-masterandnode: ssh failed"
    exit 1
  }
}

# Delete a kubernetes cluster
function kube-down {
  source "$BASE_DIR/${KUBE_CONFIG_FILE-"config-default.sh"}"

  source "$KUBE_ROOT/cluster/common.sh"
  tear_down_alive_resources

  local i=0
  for node in ${nodes}; do
    echo "kube-down: cleaning on node ${node#*@}"
    if [[ "${roles[${i}]}" == "ai" || "${roles[${i}]}" == "a" ]]; then
      ssh -t $node '
        pgrep etcd && \
        sudo -p "[sudo] password to stop master: " -- /bin/bash -c "
            service etcd stop

            rm -rf \
              /opt/bin/etcd* \
              /etc/init/etcd.conf \
              /etc/init.d/etcd \
              /etc/default/etcd

            rm -rf /infra*
        "
      ' || echo "kube-down: ssh failed when cleaning master [$node]"
    elif [[ "${roles[${i}]}" == "i" ]]; then
      ssh -t $node '
        pgrep flanneld && \
        sudo -p "[sudo] password to stop node: " -- /bin/bash -c "
            service flanneld stop
        "
      ' || echo "kube-down: ssh failed when cleaning node [$node]"
    else
      echo "unsupported role for $node"
    fi

    # Delete the files in order to generate a clean environment, so you can
    # change each node's role at next deployment.
    ssh -t $node 'sudo -- /bin/bash -c "
      rm -rf \
        /opt/bin/kube* \
        /etc/init/kube* \
        /etc/init.d/kube* \
        /etc/default/kube*

      rm -rf \
        /opt/bin/flanneld \
        /etc/init/flanneld.conf \
        /etc/init.d/flanneld \
        /etc/default/flanneld

      rm -rf ~/kube /var/lib/kubelet

    "' || echo "kube-down: ssh failed when cleaning files in node [$node]"

    ((i=i+1))
  done
}


# Perform common upgrade setup tasks
function prepare-push() {
  # Use local binaries for kube-push
  if [[ "$KUBE_VERSION" == "" ]]; then
    if [[ ! -d "$BASE_DIR/binaries" ]]; then
      echo "ERROR: No local binaries. Please check"
      exit 1
    else
      echo "Please make sure all the required local binaries are prepared"
      sleep 3
    fi
  else
    # Run build.sh to get the required release
    export KUBE_VERSION
    $BASE_DIR/build.sh
  fi
}

# Update a kubernetes master with expected release
function push-master {
  source "$BASE_DIR/${KUBE_CONFIG_FILE-"config-default.sh"}"

  if [[ ! -f "$BASE_DIR/binaries/master/kube-apiserver" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi

  setClusterInfo

  local i=0
  for node in ${nodes}; do
    if [[ "${roles[${i}]}" == "ai" || "${roles[${i}]}" == "a" ]]; then
      echo "Cleaning master ${node#*@}"
      ssh -t $node '
        sudo -p "[sudo] stop all processes in master: " -- /bin/bash -c "
          service etcd stop

          rm -rf \
            /opt/bin/etcd* \
            /etc/init/etcd.conf \
            /etc/init.d/etcd \
            /etc/default/etcd

          rm -f \
            /opt/bin/kube* \
            /etc/init/kube* \
            /etc/init.d/kube* \
            /etc/default/kube*

          rm -rf \
            /opt/bin/flanneld \
            /etc/init/flanneld.conf \
            /etc/init.d/flanneld \
            /etc/default/flanneld

          rm -rf ~/kube
        "
      ' || echo "Something failed while cleaning master"
    fi

    if [[ "${roles[${i}]}" == "a" ]]; then
      provision-master
    elif [[ "${roles[${i}]}" == "ai" ]]; then
      provision-masterandnode
    elif [[ "${roles[${i}]}" == "i" ]]; then
      ((i=i+1))
      continue
    else
      echo "ERROR: unsupported role for $node. Please check"
      exit 1
    fi

    ((i=i+1))
  done
  verify-cluster
}

# Update a kubernetes node with expected release
function push-node() {
  source "$BASE_DIR/${KUBE_CONFIG_FILE-"config-default.sh"}"

  if [[ ! -f "$BASE_DIR/binaries/minion/kubelet" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi

  setClusterInfo

  local node_ip=$1
  local existing=false
  local i=0
  for node in ${nodes}; do
    if [[ "${roles[${i}]}" == "i" && ${node#*@} == $node_ip ]]; then
      echo "Cleaning node ${node#*@}"

      ssh -t $node '
        sudo -p "[sudo] stop the all process: " -- /bin/bash -c "
          service flanneld stop

          rm -f \
            /opt/bin/kube* \
            /etc/init/kube* \
            /etc/init.d/kube* \
            /etc/default/kube*

          rm -rf \
            /opt/bin/flanneld \
            /etc/init/flanneld.conf \
            /etc/init.d/flanneld \
            /etc/default/flanneld

          rm -rf ~/kube
        "
      ' || echo "Something failed while cleaning node [$node]"
      provision-node $node
      existing=true
    elif [[ "${roles[${i}]}" == "a" || "${roles[${i}]}" == "ai" ]] && \
         [[ ${node#*@} == $node_ip ]];
    then
      echo "${node} is master node, please try ./kube-push -m instead"
    fi

    ((i=i+1))
  done

  if [[ "${existing}" == false ]]; then
    echo "node ${node_ip} does not exist"
  else
    verify-cluster
  fi
}

# Update a kubernetes cluster with expected source
function kube-push {
  prepare-push
  source "$BASE_DIR/${KUBE_CONFIG_FILE-"config-default.sh"}"

  if [[ ! -f "$BASE_DIR/binaries/master/kube-apiserver" ]]; then
    echo "There is no required release of kubernetes, please check first"
    exit 1
  fi

  #stop all the kube's process & etcd
  local i=0
  for node in $nodes; do
    echo "kube-push: Cleaning on node ${node#*@}"
    if [[ "${roles[${i}]}" == "ai" || "${roles[${i}]}" == "a" ]]; then
      ssh -t $node '
        pgrep etcd && \
        sudo -p "[sudo] password to stop master: " -- /bin/bash -c "
            service etcd stop

            rm -rf \
              /opt/bin/etcd* \
              /etc/init/etcd.conf \
              /etc/init.d/etcd \
              /etc/default/etcd
        "
      ' || echo "Something failed while cleaning etcd in master"
    elif [[ "${roles[${i}]}" == "i" ]]; then
      ssh -t $node '
        pgrep flanneld && \
        sudo -p "[sudo] password to stop node: " -- /bin/bash -c "
          service flanneld stop
        "
      ' || echo "Something failed while stopping flannel in node [$node]"
    else
      echo "unsupported role for $node"
    fi

    ssh -t $node '
      sudo -- /bin/bash -c "
        rm -f \
          /opt/bin/kube* \
          /etc/init/kube* \
          /etc/init.d/kube* \
          /etc/default/kube*

        rm -rf \
          /opt/bin/flanneld \
          /etc/init/flanneld.conf \
          /etc/init.d/flanneld \
          /etc/default/flanneld

        rm -rf ~/kube
      "
    ' || echo "Something failed while cleaning kube/flannel in node [$node]"

    ((i=i+1))
  done

  # provision all nodes, including master & nodes
  setClusterInfo

  local i=0
  for node in $nodes; do
    if [[ "${roles[${i}]}" == "a" ]]; then
      provision-master
    elif [[ "${roles[${i}]}" == "i" ]]; then
      provision-node $node
    elif [[ "${roles[${i}]}" == "ai" ]]; then
      provision-masterandnode
    else
      echo "ERROR: unsupported role for $node. Please check"
      exit 1
    fi
    ((i=i+1))
  done
  verify-cluster
}

# Perform preparations required to run e2e tests
function prepare-e2e {
  echo "Ubuntu doesn't need special preparations for e2e tests" 1>&2
}
