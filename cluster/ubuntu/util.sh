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
KUBELET_CONFIG=${KUBELET_CONFIG:-/etc/kubernetes/manifests}

MASTER_IP=""

# Assumed Vars:
#   KUBE_ROOT
function test-build-release() {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# Sanity check on $CNI_PLUGIN_CONF and $CNI_PLUGIN_EXES
function check-CNI-config() {
  if [ -z "$CNI_PLUGIN_CONF" ] && [ -n "$CNI_PLUGIN_EXES" ]; then
    echo "Warning: CNI_PLUGIN_CONF is emtpy but CNI_PLUGIN_EXES is not (it is $CNI_PLUGIN_EXES); Flannel will be used" >& 2
  elif [ -n "$CNI_PLUGIN_CONF" ] && [ -z "$CNI_PLUGIN_EXES" ]; then
    echo "Warning: CNI_PLUGIN_EXES is empty but CNI_PLUGIN_CONF is not (it is $CNI_PLUGIN_CONF); Flannel will be used" & 2
  elif [ -n "$CNI_PLUGIN_CONF" ] && [ -n "$CNI_PLUGIN_EXES" ]; then
    local problems=0
    if ! [ -r "$CNI_PLUGIN_CONF" ]; then
      echo "ERROR: CNI_PLUGIN_CONF is set to $CNI_PLUGIN_CONF but that is not a readable existing file!" >& 2
      let problems=1
    fi
    local ii=0
    for exe in $CNI_PLUGIN_EXES; do
      if ! [ -x "$exe" ]; then
        echo "ERROR: CNI_PLUGIN_EXES[$ii], which is $exe, is not an existing executable file!" >& 2
        let problems=problems+1
      fi
      let ii=ii+1
    done
    if (( problems > 0 )); then
      exit 1
    fi
  fi
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
    master_daemons=("kube-apiserver" "kube-controller-manager" "kube-scheduler")
    node_daemons=("kube-proxy" "kubelet" "docker")
  local ii=0

  for i in ${nodes}; do
      role="${roles[${ii}]}"
      if [[ "$role" =~ "a" ]]; then
          verify-daemons "$i" ${master_daemons[@]}
      fi
      if [[ "$role" =~ "i" ]]; then
          verify-daemons "$i" ${node_daemons[@]}
      fi
      if [[ "$role" =~ [^ai] ]]; then
          echo "Unsupported role for ${i}: $role."
          exit 1
      fi
      ((ii=ii+1))

    # if [ "${roles[${ii}]}" == "a" ]; then
    #   verify-master
    # elif [ "${roles[${ii}]}" == "i" ]; then
    #   verify-node "$i"
    # elif [ "${roles[${ii}]}" == "ai" ]; then
    #   verify-master
    #   verify-node "$i"
    # else
    #   echo "unsupported role for ${i}. please check"
    #   exit 1
    # fi

  done

}

function verify-daemons() {
    NODE_IP=${1#*@}
    shift
    REQUIRED_DAEMONS=$@
    # verify master has all required daemons
    echo -n "Validating that daemons ${REQUIRED_DAEMONS} run on ${NODE_IP}"
    local validated="1"
    local try_count=1
    local max_try_count=30
    until [[ "$validated" == "0" ]]; do
        validated="0"
        local daemon
        for daemon in ${REQUIRED_DAEMONS[@]}; do
            ssh $SSH_OPTS "${NODE_IP}" "pgrep -f '${daemon}'" >/dev/null 2>&1 || {
                echo -n "."
                validated="1"
                ((try_count=try_count+1))
                if [[ ${try_count} -gt ${max_try_count} ]]; then
                    echo -e "\nError: Process '${daemon}' failed to run on ${NODE_IP}, please check.\n"
                    exit 1
                fi
                sleep 2
            }
        done
    done
    echo
}

# Create ~/kube/default/etcd with proper contents.
# $1: The one IP address where the etcd leader listens.
function create-etcd-opts() {
    NODE_IP=$1
    NODE_NAME=$2
    INITIAL_CLUSTER=$3
  cat <<EOF > ~/kube/default/etcd
ETCD_OPTS="\\
 -name ${NODE_NAME} \\
 -listen-client-urls http://127.0.0.1:2379,http://127.0.0.1:4001,http://${NODE_IP}:2379,http://${NODE_IP}:4001 \\
 -advertise-client-urls http://127.0.0.1:2379,http://127.0.0.1:4001,http://${NODE_IP}:2379,http://${NODE_IP}:4001 \\
 -data-dir /var/etcd/data \\
 -initial-cluster ${INITIAL_CLUSTER} \\
 -initial-advertise-peer-urls http://${NODE_IP}:2380 \\
 -listen-peer-urls http://${NODE_IP}:2380 \\
 -initial-cluster-state new \\
"
EOF
}

# Create ~/kube/default/kube-apiserver with proper contents.
# $1: CIDR block for service addresses.
# $2: Admission Controllers to invoke in the API server.
# $3: A port range to reserve for services with NodePort visibility.
# $4: The IP address on which to advertise the apiserver to members of the cluster.
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
 --advertise-address=${4}\
 --client-ca-file=/srv/kubernetes/ca.crt\
 --tls-cert-file=/srv/kubernetes/server.cert\
 --tls-private-key-file=/srv/kubernetes/server.key"
EOF
}

# Create ~/kube/default/kube-controller-manager with proper contents.
function create-kube-controller-manager-opts() {
  cat <<EOF > ~/kube/default/kube-controller-manager
KUBE_CONTROLLER_MANAGER_OPTS="\\
 --master=127.0.0.1:8080 \\
 --root-ca-file=/srv/kubernetes/ca.crt \\
 --service-account-private-key-file=/srv/kubernetes/server.key \\
 --logtostderr=true \\
 --leader-elect=true \\
"
EOF

}

# Create ~/kube/default/kube-scheduler with proper contents.
function create-kube-scheduler-opts() {
  cat <<EOF > ~/kube/default/kube-scheduler
KUBE_SCHEDULER_OPTS=" \\
 --logtostderr=true \\
 --master=127.0.0.1:8080 \\
 --leader-elect=true \\
"
EOF

}

# Create ~/kube/default/kubelet with proper contents.
# $1: The hostname or IP address by which the kubelet will identify itself.
# $2: The one hostname or IP address at which the API server is reached (insecurely).
# $3: If non-empty then the DNS server IP to configure in each pod.
# $4: If non-empty then added to each pod's domain search list.
# $5: Pathname of the kubelet config file or directory.
# $6: If empty then flannel is used otherwise CNI is used.
function create-kubelet-opts() {
  if [ -n "$6" ] ; then
      cni_opts=" --network-plugin=cni --network-plugin-dir=/etc/cni/net.d"
  else
      cni_opts=""
  fi
  cat <<EOF > ~/kube/default/kubelet
KUBELET_OPTS="\
 --hostname-override=${1} \
 --api-servers=http://${2}:8080 \
 --logtostderr=true \
 --cluster-dns=${3} \
 --cluster-domain=${4} \
 --config=${5} \
 $cni_opts"
EOF
}

# Create ~/kube/default/kube-proxy with proper contents.
# $1: The hostname or IP address by which the node is identified.
# $2: The one hostname or IP address at which the API server is reached (insecurely).
function create-kube-proxy-opts() {
  cat <<EOF > ~/kube/default/kube-proxy
KUBE_PROXY_OPTS="\
 --hostname-override=${1} \
 --master=http://${2}:8080 \
 --logtostderr=true \
 ${3}"
EOF

}

# Create ~/kube/default/flanneld with proper contents.
# $1: The one hostname or IP address at which the etcd leader listens.
function create-flanneld-opts() {
    ETCD=$1
    IFACE=$2
  cat <<EOF > ~/kube/default/flanneld
FLANNEL_OPTS="--etcd-endpoints=http://${ETCD}:4001 \
 --ip-masq \
 --iface=${IFACE}"
EOF
}

function startup-etcd() {
    # Install ETCD on all nodes marked as master. If multiple nodes
    # are marked as master, create an ETCD cluster.
    local ii=0
    export INITIAL_CLUSTER=

    for i in ${nodes}; do
        role="${roles[${ii}]}"
        if [[ "$role" =~ "a" ]]; then
            NODE_IP=${i#*@}
            NODE_NAME=infra${ii}
            if [ -n "${INITIAL_CLUSTER}" ]; then
                INITIAL_CLUSTER="${INITIAL_CLUSTER},${NODE_NAME}=http://${NODE_IP}:2380"
            else
                INITIAL_CLUSTER="${NODE_NAME}=http://${NODE_IP}:2380"
            fi
        fi
        ((ii=ii + 1))
    done

    ii=0
    for i in ${nodes}; do
        role="${roles[${ii}]}"
        if [[ "$role" =~ "a" ]]; then
            NODE_NAME=infra${ii}
            provision-etcd "$i" "${NODE_NAME}"
        fi
        ((ii=ii + 1))
    done
}

function detect-master {
    # Nothing to do, we've already detected the master. We provide a
    # definition for this function here to prevent the default one in
    # kube-util.sh from outputting annoying information.
    true
}

# Instantiate a kubernetes cluster on ubuntu
function kube-up() {
  export KUBE_CONFIG_FILE=${KUBE_CONFIG_FILE:-${KUBE_ROOT}/cluster/ubuntu/config-default.sh}
  source "${KUBE_CONFIG_FILE}"

  # downloading tarball release
  "${KUBE_ROOT}/cluster/ubuntu/download-release.sh"

  # Fetch the hacked easyrsa that make-ca-cert.sh will use
  curl -L -O https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz > /dev/null 2>&1

  if ! check-CNI-config; then
    return
  fi

  startup-etcd

  # The IP address of one of the masters.
  MASTER_IP=

  # Install master nodes first
  local ii=0
  for i in ${nodes}; do
      role="${roles[${ii}]}"
      if [[ "$role" =~ "a" ]]; then
          provision-master "$i"
          MASTER_IP=${i#*@}
          export KUBE_MASTER=${i}
          export KUBE_MASTER_IP=${i#*@}
      fi
      ((ii=ii+1))
  done

  # Install minion nodes second
  local ii=0
  for i in ${nodes}; do
      role="${roles[${ii}]}"
      if [[ "$role" =~ "i" ]]; then
          if [[ "$role" =~ "a" ]]; then
              is_master=true
          else
              is_master=false
          fi
          provision-node "$i" "$is_master"
      fi
      ((ii=ii+1))
  done

  wait

  export KUBECTL_PATH="${KUBE_ROOT}/cluster/ubuntu/binaries/kubectl"
  verify-cluster
  export CONTEXT="ubuntu"
  export KUBE_SERVER="http://${MASTER_IP}:8080"

  source "${KUBE_ROOT}/cluster/common.sh"

  # set kubernetes user and password
  load-or-gen-kube-basicauth

  create-kubeconfig
}

function provision-master() {
    NODE_IP=${1#*@}

  echo -e "\nDeploying MASTER on machine ${NODE_IP}"

  ssh $SSH_OPTS "$NODE_IP" "mkdir -p ~/kube/default"

  # copy the binaries and scripts to the ~/kube directory on the master
  scp -r $SSH_OPTS \
    saltbase/salt/generate-cert/make-ca-cert.sh \
    easy-rsa.tar.gz \
    ubuntu/reconfDocker.sh \
    "${KUBE_CONFIG_FILE}" \
    ubuntu/util.sh \
    ubuntu/master/* \
    ubuntu/binaries/master/ \
    "${NODE_IP}:~/kube"

  if [ -z "$CNI_PLUGIN_CONF" ] || [ -z "$CNI_PLUGIN_EXES" ]; then
    # Flannel is being used: copy the flannel binaries and scripts, set reconf flag
    scp -r $SSH_OPTS ubuntu/master-flannel/* "${NODE_IP}:~/kube"
    NEED_RECONFIG_DOCKER=true
    SERVICE_STARTS="service flanneld stop || true
                    killall -q flanneld || true
                    service flanneld start"
  else
    # CNI is being used: set reconf flag
    NEED_RECONFIG_DOCKER=false
  fi

  EXTRA_SANS=(
    IP:$NODE_IP
    IP:${SERVICE_CLUSTER_IP_RANGE%.*}.1
    DNS:kubernetes
    DNS:kubernetes.default
    DNS:kubernetes.default.svc
    DNS:kubernetes.default.svc.cluster.local
  )

  EXTRA_SANS=$(echo "${EXTRA_SANS[@]}" | tr ' ' ,)

  BASH_DEBUG_FLAGS=""
  if [[ "$DEBUG" == "true" ]] ; then
    BASH_DEBUG_FLAGS="set -x"
  fi

  # remote login to MASTER and configue k8s master
  ssh $SSH_OPTS -t "${NODE_IP}" "
    set +e
    set +o pipefail
    ${BASH_DEBUG_FLAGS}
    source ~/kube/util.sh

    create-kube-apiserver-opts \
      '${SERVICE_CLUSTER_IP_RANGE}' \
      '${ADMISSION_CONTROL}' \
      '${SERVICE_NODE_PORT_RANGE}' \
      '${NODE_IP}'
    create-kube-controller-manager-opts
    create-kube-scheduler-opts
    create-flanneld-opts '127.0.0.1' '${NODE_IP}'
    FLANNEL_OTHER_NET_CONFIG='${FLANNEL_OTHER_NET_CONFIG}' sudo -E -p '[sudo] password to start master: ' -- /bin/bash -ce '
      ${BASH_DEBUG_FLAGS}

      cp ~/kube/default/* /etc/default/
      cp ~/kube/init_conf/* /etc/init/
      cp ~/kube/init_scripts/* /etc/init.d/

      groupadd -f -r kube-cert
      ${PROXY_SETTING} DEBUG='${DEBUG}' ~/kube/make-ca-cert.sh \"${NODE_IP}\" \"${EXTRA_SANS}\"
      mkdir -p /opt/bin/
      cp \`ls ~/kube/master/* | grep -v etcd\` /opt/bin/

      ${SERVICE_STARTS}

      service kube-apiserver start
      service kube-controller-manager start
      service kube-scheduler start

      if ${NEED_RECONFIG_DOCKER}; then FLANNEL_NET=\"${FLANNEL_NET}\" KUBE_CONFIG_FILE=\"${KUBE_CONFIG_FILE}\" DOCKER_OPTS=\"${DOCKER_OPTS}\" ~/kube/reconfDocker.sh a; fi
      '" || {
      echo "Deploying MASTER on machine ${NODE_IP} failed"
      exit 1
    }
}

function provision-node() {
    NODE_IP=${1#*@}
    IS_MASTER="$2"
  echo -e "\nDeploying NODE on machine ${NODE_IP}"

  ssh $SSH_OPTS $1 "mkdir -p ~/kube/default"

  # copy the binaries and scripts to the ~/kube directory on the node
  scp -r $SSH_OPTS \
    "${KUBE_CONFIG_FILE}" \
    ubuntu/util.sh \
    ubuntu/reconfDocker.sh \
    ubuntu/minion/* \
    ubuntu/master/* \
    ubuntu/binaries/master \
    ubuntu/binaries/minion \
    "${1}:~/kube"

  if [ -z "$CNI_PLUGIN_CONF" ] || [ -z "$CNI_PLUGIN_EXES" ]; then
    # Prep for Flannel use: copy the flannel binaries and scripts, set reconf flag
    scp -r $SSH_OPTS ubuntu/minion-flannel/* "${1}:~/kube"
    SERVICE_STARTS="service flanneld stop || true
                    killall -q flanneld || true
                    service flanneld start"
    NEED_RECONFIG_DOCKER=true
    CNI_PLUGIN_CONF=''

  else
    # Prep for CNI use: copy the CNI config and binaries, adjust upstart config, set reconf flag
    ssh $SSH_OPTS "${1}" "rm -rf tmp-cni; mkdir -p tmp-cni/exes tmp-cni/conf"
    scp    $SSH_OPTS "$CNI_PLUGIN_CONF" "${1}:tmp-cni/conf/"
    scp -p $SSH_OPTS  $CNI_PLUGIN_EXES  "${1}:tmp-cni/exes/"
    ssh $SSH_OPTS -t "${1}" '
      sudo -p "[sudo] password to prep node %h: " -- /bin/bash -ce "
        mkdir -p /opt/cni/bin /etc/cni/net.d
        cp ~$(id -un)/tmp-cni/conf/* /etc/cni/net.d/
        cp --preserve=mode ~$(id -un)/tmp-cni/exes/* /opt/cni/bin/
        '"sed -i.bak -e 's/start on started flanneld/start on started ${CNI_KUBELET_TRIGGER}/' -e 's/stop on stopping flanneld/stop on stopping ${CNI_KUBELET_TRIGGER}/' "'~$(id -un)/kube/init_conf/kubelet.conf
        '"sed -i.bak -e 's/start on started flanneld/start on started networking/' -e 's/stop on stopping flanneld/stop on stopping networking/' "'~$(id -un)/kube/init_conf/kube-proxy.conf
        "'
    SERVICE_STARTS='service kubelet    start
                    service kube-proxy start'
    NEED_RECONFIG_DOCKER=false
  fi

  BASH_DEBUG_FLAGS=""
  if [[ "$DEBUG" == "true" ]] ; then
    BASH_DEBUG_FLAGS="set -x"
  fi

  # remote login to node and configue k8s node
  echo Will run script with flannel: ${SERVICE_STARTS}
  ssh $SSH_OPTS -t "$1" "
    set +e
    set +o pipefail
    ${BASH_DEBUG_FLAGS}
    source ~/kube/util.sh

    create-kube-controller-manager-opts
    create-kube-scheduler-opts
    create-kubelet-opts \
      '${NODE_IP}' \
      '${MASTER_IP}' \
      '${DNS_SERVER_IP}' \
      '${DNS_DOMAIN}' \
      '${KUBELET_CONFIG}' \
      '${CNI_PLUGIN_CONF}'
    create-kube-proxy-opts \
      '${NODE_IP}' \
      '${MASTER_IP}' \
      '${KUBE_PROXY_EXTRA_OPTS}'

    if [ '${IS_MASTER}' == 'true' ]; then
        create-flanneld-opts '127.0.0.1' '${NODE_IP}'
    else
        create-flanneld-opts '${MASTER_IP}' '${NODE_IP}'
    fi

    sudo -E -p '[sudo] password to start node: ' -- /bin/bash -ce '
      ${BASH_DEBUG_FLAGS}
      cp ~/kube/default/* /etc/default/
      cp ~/kube/init_conf/* /etc/init/
      cp ~/kube/init_scripts/* /etc/init.d/
      mkdir -p /opt/bin/
      cp ~/kube/minion/* /opt/bin || true
      ${SERVICE_STARTS}
      if ${NEED_RECONFIG_DOCKER}; then KUBE_CONFIG_FILE=\"${KUBE_CONFIG_FILE}\" DOCKER_OPTS=\"${DOCKER_OPTS}\" ~/kube/reconfDocker.sh i; fi
      '" || {
      echo "Deploying NODE on machine ${NODE_IP} failed"
      exit 1
  }
}

function provision-etcd() {
    NODE_IP=${1#*@}
    NODE_NAME="$2"
    echo -e "\nProvisioning ETCD on machine ${NODE_IP}"
  ssh $SSH_OPTS $NODE_IP "mkdir -p ~/kube/etcd"
  ssh $SSH_OPTS $NODE_IP "mkdir -p ~/kube/default"
  ssh $SSH_OPTS $NODE_IP "mkdir -p ~/kube/init_conf"
  scp -r $SSH_OPTS \
      ubuntu/binaries/master/etcd* \
      "${NODE_IP}:~/kube/etcd"

  scp -r $SSH_OPTS \
      ubuntu/master/init_conf/etcd.conf \
      "${NODE_IP}:~/kube/init_conf"

  scp -r $SSH_OPTS \
      "${KUBE_CONFIG_FILE}" \
      ubuntu/util.sh \
      "${NODE_IP}:~/kube"

  BASH_DEBUG_FLAGS=""
  if [[ "$DEBUG" == "true" ]] ; then
    BASH_DEBUG_FLAGS="set -x"
  fi

  ssh $SSH_OPTS -t "$NODE_IP" "
    set +e
    set +o pipefail
    ${BASH_DEBUG_FLAGS}
    source ~/kube/util.sh
    cp ~/kube/init_conf/* /etc/init/
    mkdir -p /opt/bin/
    service etcd stop || true
    killall -q etcd || true
    cp ~/kube/etcd/* /opt/bin/
    rm -rf /var/etcd/data
    create-etcd-opts ${NODE_IP} ${NODE_NAME} \"${INITIAL_CLUSTER}\"
    cp ~/kube/default/* /etc/default/
    service etcd start
  " || {
      echo "Deploying ETCD on machine ${NODE_IP} failed."
      exit 1
  }
}

# check whether kubelet has torn down all of the pods
function check-pods-torn-down() {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local attempt=0
  while [[ ! -z "$(kubectl get pods --show-all --all-namespaces| tail -n +2)" ]]; do
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

  export KUBE_CONFIG_FILE=${KUBE_CONFIG_FILE:-${KUBE_ROOT}/cluster/ubuntu/config-default.sh}
  source "${KUBE_CONFIG_FILE}"

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
            set +o pipefail
	    service etcd stop
	    killall -q etcd

            rm -rf \
              /opt/bin/etcd* \
              /etc/init/etcd.conf \
              /etc/init.d/etcd \
              /etc/default/etcd \
              /var/etcd/data

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
  export KUBE_CONFIG_FILE=${KUBE_CONFIG_FILE:-${KUBE_ROOT}/cluster/ubuntu/config-default.sh}
  source "${KUBE_CONFIG_FILE}"

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
  export KUBE_CONFIG_FILE=${KUBE_CONFIG_FILE:-${KUBE_ROOT}/cluster/ubuntu/config-default.sh}
  source "${KUBE_CONFIG_FILE}"

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
  export KUBE_CONFIG_FILE=${KUBE_CONFIG_FILE:-${KUBE_ROOT}/cluster/ubuntu/config-default.sh}
  source "${KUBE_CONFIG_FILE}"

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
