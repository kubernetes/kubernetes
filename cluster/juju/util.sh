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


set -o errexit
set -o nounset
set -o pipefail
#set -o xtrace

UTIL_SCRIPT=$(readlink -m "${BASH_SOURCE}")
JUJU_PATH=$(dirname ${UTIL_SCRIPT})
KUBE_ROOT=$(readlink -m ${JUJU_PATH}/../../)
# Use the config file specified in $KUBE_CONFIG_FILE, or config-default.sh.
source "${JUJU_PATH}/${KUBE_CONFIG_FILE-config-default.sh}"
source ${JUJU_PATH}/prereqs/ubuntu-juju.sh
export JUJU_REPOSITORY=${JUJU_PATH}/charms
#KUBE_BUNDLE_URL='https://raw.githubusercontent.com/whitmo/bundle-kubernetes/master/bundles.yaml'
KUBE_BUNDLE_PATH=${JUJU_PATH}/bundles/local.yaml

# Build the binaries on the local system and copy the binaries to the Juju charm.
function build-local() {
    local targets=(
        cmd/kube-proxy \
        cmd/kube-apiserver \
        cmd/kube-controller-manager \
        cmd/kubelet \
        plugin/cmd/kube-scheduler \
        cmd/kubectl \
        test/e2e/e2e.test \
    )
    # Make a clean environment to avoid compiler errors.
    make clean
    # Build the binaries locally that are used in the charms.
    make all WHAT="${targets[*]}"
    local OUTPUT_DIR=_output/local/bin/linux/amd64
    mkdir -p cluster/juju/charms/trusty/kubernetes-master/files/output
    # Copy the binaries from the output directory to the charm directory.
    cp -v $OUTPUT_DIR/* cluster/juju/charms/trusty/kubernetes-master/files/output
}

function detect-master() {
    local kubestatus
    # Capturing a newline, and my awk-fu was weak - pipe through tr -d
    kubestatus=$(juju status --format=oneline kubernetes-master | grep kubernetes-master/0 | awk '{print $3}' | tr -d "\n")
    export KUBE_MASTER_IP=${kubestatus}
    export KUBE_SERVER=http://${KUBE_MASTER_IP}:8080
}

function detect-nodes() {
    # Run the Juju command that gets the minion private IP addresses.
    local ipoutput
    ipoutput=$(juju run --service kubernetes "unit-get private-address" --format=json)
    # [
    # {"MachineId":"2","Stdout":"192.168.122.188\n","UnitId":"kubernetes/0"},
    # {"MachineId":"3","Stdout":"192.168.122.166\n","UnitId":"kubernetes/1"}
    # ]

    # Strip out the IP addresses
    export KUBE_NODE_IP_ADDRESSES=($(${JUJU_PATH}/return-node-ips.py "${ipoutput}"))
    # echo "Kubernetes minions: " ${KUBE_NODE_IP_ADDRESSES[@]} 1>&2
    export NUM_NODES=${#KUBE_NODE_IP_ADDRESSES[@]}
}

function get-password() {
  export KUBE_USER=admin
  # Get the password from the basic-auth.csv file on kubernetes-master.
  export KUBE_PASSWORD=$(juju run --unit kubernetes-master/0 "cat /srv/kubernetes/basic-auth.csv" | grep ${KUBE_USER} | cut -d, -f1)
}

function kube-up() {
    build-local
    if [[ -d "~/.juju/current-env" ]]; then
        juju quickstart -i --no-browser
    else
        juju quickstart --no-browser
    fi
    # The juju-deployer command will deploy the bundle and can be run
    # multiple times to continue deploying the parts that fail.
    juju deployer -c ${KUBE_BUNDLE_PATH}

    source "${KUBE_ROOT}/cluster/common.sh"
    get-password

    # Sleep due to juju bug http://pad.lv/1432759
    sleep-status
    detect-master
    detect-nodes

    local prefix=$RANDOM
    export KUBE_CERT="/tmp/${prefix}-kubecfg.crt"
    export KUBE_KEY="/tmp/${prefix}-kubecfg.key"
    export CA_CERT="/tmp/${prefix}-kubecfg.ca"
    export CONTEXT="juju"

    # Copy the cert and key to this machine.
    (
      umask 077
      juju scp kubernetes-master/0:/srv/kubernetes/apiserver.crt ${KUBE_CERT}
      juju run --unit kubernetes-master/0 'chmod 644 /srv/kubernetes/apiserver.key'
      juju scp kubernetes-master/0:/srv/kubernetes/apiserver.key ${KUBE_KEY}
      juju run --unit kubernetes-master/0 'chmod 600 /srv/kubernetes/apiserver.key'
      cp ${KUBE_CERT} ${CA_CERT}

      create-kubeconfig
    )
}

function kube-down() {
    local force="${1-}"
    # Remove the binary files from the charm directory.
    rm -rf cluster/juju/charms/trusty/kubernetes-master/files/output/
    local jujuenv
    jujuenv=$(cat ~/.juju/current-environment)
    juju destroy-environment ${jujuenv} ${force} || true
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

    jujustatus=$(juju status kubernetes-master --format=oneline)
    if [[ $jujustatus == *"started"* ]];
    then
        return
    fi

    while [[ $i < $maxtime && $jujustatus != *"started"* ]]; do
        sleep 15
        i+=15
        jujustatus=$(juju status kubernetes-master --format=oneline)
    done

    # sleep because we cannot get the status back of where the minions are in the deploy phase
    # thanks to a generic "started" state and our service not actually coming online until the
    # minions have received the binary from the master distribution hub during relations
    echo "Sleeping an additional minute to allow the cluster to settle" 1>&2
    sleep 60
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
