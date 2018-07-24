#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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


set -o errexit
set -o nounset
set -o pipefail
#set -o xtrace

UTIL_SCRIPT=$(readlink -m "${BASH_SOURCE}")
JUJU_PATH=$(dirname ${UTIL_SCRIPT})
KUBE_ROOT=$(readlink -m ${JUJU_PATH}/../../)
# Use the config file specified in $KUBE_CONFIG_FILE, or config-default.sh.
source "${JUJU_PATH}/${KUBE_CONFIG_FILE-config-default.sh}"
# This attempts installation of Juju - This really needs to support multiple
# providers/distros - but I'm super familiar with ubuntu so assume that for now.
source "${JUJU_PATH}/prereqs/ubuntu-juju.sh"
export JUJU_REPOSITORY="${JUJU_PATH}/charms"
KUBE_BUNDLE_PATH="${JUJU_PATH}/bundles/local.yaml"
# The directory for the kubectl binary, this is one of the paths in kubectl.sh.
KUBECTL_DIR="${KUBE_ROOT}/platforms/linux/amd64"


function build-local() {
    # This used to build the kubernetes project. Now it rebuilds the charm(s)
    # living in `cluster/juju/layers`

    charm build ${JUJU_PATH}/layers/kubernetes -o $JUJU_REPOSITORY -r --no-local-layers
}

function detect-master() {
    local kubestatus

    # Capturing a newline, and my awk-fu was weak - pipe through tr -d
    kubestatus=$(juju status --format=oneline kubernetes | grep ${KUBE_MASTER_NAME} | awk '{print $3}' | tr -d "\n")
    export KUBE_MASTER_IP=${kubestatus}
    export KUBE_SERVER=https://${KUBE_MASTER_IP}:6433

}

function detect-nodes() {
    # Run the Juju command that gets the minion private IP addresses.
    local ipoutput
    ipoutput=$(juju run --application kubernetes "unit-get private-address" --format=json)
    # [
    # {"MachineId":"2","Stdout":"192.168.122.188\n","UnitId":"kubernetes/0"},
    # {"MachineId":"3","Stdout":"192.168.122.166\n","UnitId":"kubernetes/1"}
    # ]

    # Strip out the IP addresses
    export KUBE_NODE_IP_ADDRESSES=($(${JUJU_PATH}/return-node-ips.py "${ipoutput}"))
    # echo "Kubernetes minions: " ${KUBE_NODE_IP_ADDRESSES[@]} 1>&2
    export NUM_NODES=${#KUBE_NODE_IP_ADDRESSES[@]}
}

function kube-up() {
    build-local

    # Replace the charm directory in the bundle.
    sed "s|__CHARM_DIR__|${JUJU_REPOSITORY}|" < ${KUBE_BUNDLE_PATH}.base > ${KUBE_BUNDLE_PATH}

    # The juju-deployer command will deploy the bundle and can be run
    # multiple times to continue deploying the parts that fail.
    juju deploy ${KUBE_BUNDLE_PATH}

    source "${KUBE_ROOT}/cluster/common.sh"

    # Sleep due to juju bug http://pad.lv/1432759
    sleep-status
    detect-master
    detect-nodes

    # Copy kubectl, the cert and key to this machine from master.
    (
      umask 077
      mkdir -p ${KUBECTL_DIR}
      juju scp ${KUBE_MASTER_NAME}:kubectl_package.tar.gz ${KUBECTL_DIR}
      tar xfz ${KUBECTL_DIR}/kubectl_package.tar.gz -C ${KUBECTL_DIR}
    )
    # Export the location of the kubectl configuration file.
    export KUBECONFIG="${KUBECTL_DIR}/kubeconfig"
}

function kube-down() {
    local force="${1-}"
    local jujuenv
    jujuenv=$(juju switch)
    juju destroy-model ${jujuenv} ${force} || true
    # Clean up the generated charm files.
    rm -rf ${KUBE_ROOT}/cluster/juju/charms
    # Clean up the kubectl binary and config file.
    rm -rf ${KUBECTL_DIR}
}

function prepare-e2e() {
  echo "prepare-e2e() The Juju provider does not need any preparations for e2e." 1>&2
}

function sleep-status() {
    local i
    local maxtime
    local jujustatus
    i=0
    maxtime=900
    jujustatus=''
    echo "Waiting up to 15 minutes to allow the cluster to come online... wait for it..." 1>&2

    while [[ $i < $maxtime && -z $jujustatus ]]; do
      sleep 15
      i=$((i + 15))
      jujustatus=$(${JUJU_PATH}/identify-leaders.py)
      export KUBE_MASTER_NAME=${jujustatus}
    done

}

# Execute prior to running tests to build a release if required for environment.
function test-build-release {
    echo "test-build-release() " 1>&2
}

# Execute prior to running tests to initialize required structure. This is
# called from hack/e2e.go only when running -up.
function test-setup {
  "${KUBE_ROOT}/cluster/kube-up.sh"
}

# Execute after running tests to perform any required clean-up. This is called
# from hack/e2e.go
function test-teardown() {
    kube-down "-y"
}

# Verify the prerequisites are statisfied before running.
function verify-prereqs() {
    gather_installation_reqs
}
