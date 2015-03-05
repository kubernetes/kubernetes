#!/bin/bash

# Copyright 2014 Canonical LTD. All rights reserved.
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

# If you find any bugs within this script - please file bugs against the
# Kubernetes Juju Charms project - located here: https://github.com/whitmo/bundle-kubernetes


source $KUBE_ROOT/cluster/juju/prereqs/ubuntu-juju.sh
kube_bundle_url='https://raw.githubusercontent.com/whitmo/bundle-kubernetes/master/bundles.yaml'
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
    local JUJUSTATUS=$(juju status kubernetes-master --format=oneline)
    if [[ -z "$JUJUSTATUS" ]]; then
        if [[ -d "~/.juju/current-env" ]]; then
            juju quickstart -i -e $kube_jujuenv --no-browser -i $kube_bundle_url
            # sleeping because of juju bug #
            sleep 120
        else
            juju quickstart --no-browser $kube_bundle_url
            # sleeping because of juju bug #
            sleep 120
        fi
    fi
    sleep-status
}


function detect-master() {
    foo=$(juju status --format=oneline kubernetes-master | cut -d' ' -f3)
    export KUBE_MASTER_IP=`echo -n $foo`
    export KUBE_MASTER=$KUBE_MASTER_IP:8080
    export KUBERNETES_MASTER=$KUBE_MASTER

   }

function detect-minions(){
    KUBE_MINION_IP_ADDRESSES=($(juju run --service kubernetes "unit-get private-address" --format=yaml | grep Stdout | cut -d "'" -f 2))
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
    local i=0
    local maxtime=900
    local JUJUSTATUS=$(juju status kubernetes-master --format=oneline)
    echo "Waiting up to 15 minutes to allow the cluster to come online... wait for it..."

    while [[ $i < $maxtime ]] && [[ $JUJUSTATUS != *"started"* ]]; do
        sleep 30
        i+=30
        JUJUSTATUS=$(juju status kubernetes-master --format=oneline)
    done

    # sleep because we cannot get the status back of where the minions are in the deploy phase
    # thanks to a generic "started" state and our service not actually coming online until the
    # minions have recieved the binary from the master distribution hub during relations
    echo "Sleeping an additional minute to allow the cluster to settle"
    sleep 60

}

