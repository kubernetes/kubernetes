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

source $KUBE_ROOT/cluster/juju/prereqs/ubuntu-juju.sh
KUBE_BUNDLE_URL='https://raw.githubusercontent.com/whitmo/bundle-kubernetes/master/bundles.yaml'
function verify-prereqs() {
    gather_installation_reqs
}

function get-password() {
    echo "TODO: Assign username/password security"
}

function kube-up() {
    # If something were to happen that I'm not accounting for, do not
    # punish the user by making them tear things down. In a perfect world
    # quickstart should handle this situation, so be nice in the meantime
    local envstatus
    envstatus=$(juju status kubernetes-master --format=oneline)

    if [[ "" == "${envstatus}" ]]; then
        if [[ -d "~/.juju/current-env" ]]; then
            juju quickstart -i --no-browser -i $KUBE_BUNDLE_URL
        else
            juju quickstart --no-browser ${KUBE_BUNDLE_URL}
        fi
        sleep 60
    fi
    # Sleep due to juju bug http://pad.lv/1432759
    sleep-status
}


function detect-master() {
    local kubestatus
    # Capturing a newline, and my awk-fu was weak - pipe through tr -d
    kubestatus=$(juju status --format=oneline kubernetes-master | awk '{print $3}' | tr -d "\n")
    export KUBE_MASTER_IP=${kubestatus}
    export KUBE_MASTER=$KUBE_MASTER_IP:8080
    export KUBERNETES_MASTER=$KUBE_MASTER

   }

function detect-minions(){
    # Strip out the components except for STDOUT return
    # and trim out the single quotes to build an array of minions
    #
    # Example Output:
    #- MachineId: "10"
    #  Stdout: '10.197.55.232
    #'
    # UnitId: kubernetes/0
    # - MachineId: "11"
    # Stdout: '10.202.146.124
    # '
    #  UnitId: kubernetes/1
 
    KUBE_MINION_IP_ADDRESSES=($(juju run --service kubernetes \
        "unit-get private-address" --format=yaml \
        | awk '/Stdout/ {gsub(/'\''/,""); print $2}'))
    NUM_MINIONS=${#KUBE_MINION_IP_ADDRESSES[@]}
    MINION_NAMES=$KUBE_MINION_IP_ADDRESSES
}

function setup-logging-firewall(){
    echo "TODO: setup logging and firewall rules"
}

function teardown-logging-firewall(){
    echo "TODO: teardown logging and firewall rules"
}


function sleep-status(){
    local i
    local maxtime
    local jujustatus
    i=0
    maxtime=900
    jujustatus=''
    echo "Waiting up to 15 minutes to allow the cluster to come online... wait for it..."
    while [[ $i < $maxtime && $jujustatus != *"started"* ]]; do
        jujustatus=$(juju status kubernetes-master --format=oneline)
        sleep 30
        i+=30
    done

    # sleep because we cannot get the status back of where the minions are in the deploy phase
    # thanks to a generic "started" state and our service not actually coming online until the
    # minions have recieved the binary from the master distribution hub during relations
    echo "Sleeping an additional minute to allow the cluster to settle"
    sleep 60
}

