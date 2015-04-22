#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

UTIL_SCRIPT=$(readlink -m "${BASH_SOURCE}")
JUJU_PATH=$(dirname ${UTIL_SCRIPT})
source ${JUJU_PATH}/prereqs/ubuntu-juju.sh
export JUJU_REPOSITORY=${JUJU_PATH}/charms
#KUBE_BUNDLE_URL='https://raw.githubusercontent.com/whitmo/bundle-kubernetes/master/bundles.yaml'
KUBE_BUNDLE_PATH=${JUJU_PATH}/bundles/local.yaml

function verify-prereqs() {
    gather_installation_reqs
}

function get-password() {
    echo "TODO: Assign username/password security"
}

function kube-up() {
    if [[ -d "~/.juju/current-env" ]]; then
        juju quickstart -i --no-browser
    else
        juju quickstart --no-browser
    fi
    # The juju-deployer command will deploy the bundle and can be run
    # multiple times to continue deploying the parts that fail.
    juju deployer -c ${KUBE_BUNDLE_PATH}
    # Sleep due to juju bug http://pad.lv/1432759
    sleep-status
    detect-master
    detect-minions
}

function kube-down() {
    local jujuenv
    jujuenv=$(cat ~/.juju/current-environment)
    juju destroy-environment $jujuenv
}

function detect-master() {
    local kubestatus
    # Capturing a newline, and my awk-fu was weak - pipe through tr -d
    kubestatus=$(juju status --format=oneline kubernetes-master | awk '{print $3}' | tr -d "\n")
    export KUBE_MASTER_IP=${kubestatus}
    export KUBE_MASTER=${KUBE_MASTER_IP}
    export KUBERNETES_MASTER=http://${KUBE_MASTER}:8080
    echo "Kubernetes master: " ${KUBERNETES_MASTER}
}

function detect-minions() {
    # Run the Juju command that gets the minion private IP addresses.
    local ipoutput
    ipoutput=$(juju run --service kubernetes "unit-get private-address" --format=json)
    echo $ipoutput
    # Strip out the IP addresses
    #
    # Example Output:
    #- MachineId: "10"
    #  Stdout: |
    #    10.197.55.232
    # UnitId: kubernetes/0
    # - MachineId: "11"
    # Stdout: |
    #    10.202.146.124
    #  UnitId: kubernetes/1
    export KUBE_MINION_IP_ADDRESSES=($(${JUJU_PATH}/return-node-ips.py "${ipoutput}"))
    echo "Kubernetes minions:  " ${KUBE_MINION_IP_ADDRESSES[@]}
    export NUM_MINIONS=${#KUBE_MINION_IP_ADDRESSES[@]}
    export MINION_NAMES=$KUBE_MINION_IP_ADDRESSES
}

function setup-logging-firewall() {
    echo "TODO: setup logging and firewall rules"
}

function teardown-logging-firewall() {
    echo "TODO: teardown logging and firewall rules"
}

function sleep-status() {
    local i
    local maxtime
    local jujustatus
    i=0
    maxtime=900
    jujustatus=''
    echo "Waiting up to 15 minutes to allow the cluster to come online... wait for it..."

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
    # minions have recieved the binary from the master distribution hub during relations
    echo "Sleeping an additional minute to allow the cluster to settle"
    sleep 60
}
