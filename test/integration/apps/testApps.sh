#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

#
# Api Server Set of functions
# In the integration test suites we create real api servers
#
apiserver::build () {
  #Make sure the binary is already available
  if [ -e ${KUBE_API_SERVER} ]; then
    kube::log::status "APi Server binary  is available here ${KUBE_API_SERVER}"
  else
    pushd ${KUBE_ROOT}
    kube::log::status "APi Server binary is not available, building now..."
    make WHAT="cmd/kube-apiserver"
    popd
  fi
}

apiserver::start () {

    # First try to build it, if not available
    apiserver::build

    kube::log::status "Starting Api server..."
##--authorization-mode=Node,RBAC \
    ${KUBE_API_SERVER} \
    --etcd-servers=http://127.0.0.1:2379 \
    --service-cluster-ip-range=10.96.0.0/12 \
    --insecure-bind-address=0.0.0.0 \
    --insecure-port=8585 \
    --cert-dir=${PWD}/cert \
    --service-account-key-file=${PWD}/cert/sa.pub \
    --admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota > ./logs/kube_api_server.log 2>&1 &

    API_SRV_PID=$!

    kube::log::status "Started Api server...PID=${API_SRV_PID}"
}

apiserver::stop () {
    kube::log::status "Stoping Api server..."

    kill -9 "${API_SRV_PID-}" >/dev/null 2>&1 || :
    wait "${API_SRV_PID-}" >/dev/null 2>&1 || :

    kube::log::status "Stopped Api server..."
}


#
# Scheduler Set of functions
# In the integration test suites we create real api servers
#
scheduler::build () {
  #Make sure the binary is already available
  if [ -e ${KUBE_SCHEDULER} ]; then
    kube::log::status "Scheduler binary  is available here ${KUBE_SCHEDULER}"
  else
    pushd ${KUBE_ROOT}
    kube::log::status "Scheduler binary is not available, building now..."
    make WHAT="plugin/cmd/kube-scheduler"
    popd
  fi
}

scheduler::start () {

    # First try to build it, if not available
    scheduler::build

    kube::log::status "Starting Scheduler..."

    ${KUBE_SCHEDULER} \
    --master=http://0.0.0.0:8585 > ./logs/kube_sceduler.log 2>&1 &

    SCHED_PID=$!

    kube::log::status "Started Scheduler...PID=${SCHED_PID}"
}

scheduler::stop () {
    kube::log::status "Stoping Scheduler..."

    kill -9 "${SCHED_PID-}" >/dev/null 2>&1 || :
    wait "${SCHED_PID-}" >/dev/null 2>&1 || :

    kube::log::status "Stopped Scheduler..."
}

#
# Controller Manager of functions
# In the integration test suites we create real api servers
#
controllermanager::build () {
  #Make sure the binary is already available
  if [ -e ${KUBE_CONTROLLER_MANAGER} ]; then
    kube::log::status "Controller Manager binary  is available here ${KUBE_CONTROLLER_MANAGER}"
  else
    pushd ${KUBE_ROOT}
    kube::log::status "Controller Manager is not available, building now..."
    make WHAT="cmd/kube-controller-manager"
    popd
  fi
}

controllermanager::start () {

    # First try to build it, if not available
    controllermanager::build

    kube::log::status "Starting Controller Manager..."

    ${KUBE_CONTROLLER_MANAGER} \
    --service-account-private-key-file=${PWD}/cert/sa.key \
    --master=http://0.0.0.0:8585 > ./logs/kube_controller_manager.log 2>&1 &

    CM_PID=$!

    kube::log::status "Started Controller Manager...PID=${CM_PID}"
}

controllermanager::stop () {
    kube::log::status "Stoping Controller Manager..."

    kill -9 "${CM_PID-}" >/dev/null 2>&1 || :
    wait "${CM_PID-}" >/dev/null 2>&1 || :

    kube::log::status "Stopped Controller Manager..."
}

cleanup() {
  kube::log::status "App Integration test cleanup started"
  controllermanager::stop
  scheduler::stop
  apiserver::stop
  kube::etcd::cleanup
  kube::log::status "App Integration test cleanup completed"

}

function configure-kubectl {
  kube::log::status "Setting cluster config"
  "${kubectl}" config set-cluster app-testing --server="http://localhost:8585" --insecure-skip-tls-verify=true
  "${kubectl}" config set-context app-testing --cluster=app-testing
  "${kubectl}" config use-context app-testing
}

# Remove component binaries such that they will be re-built
function remove-binaries {

if [ ${REBUILD_OPT}  == "true" ]; then
    kube::log::status "Removing Binaries, they will be rebuilt automatically"
    rm -vf ${KUBE_API_SERVER}
    rm -vf ${KUBE_SCHEDULER}
    rm -vf ${KUBE_CONTROLLER_MANAGER}

fi

}

# Print Help Message
function print-help {
    echo "This script can have following optional arguments..."
    echo "rebuild - removes the binaries and rebuild api-server, scheduler and controller-manager"
    echo "dev - In case of failures, leaves the components running so that it can be interacted with kubctl and investigated"
    echo "help - Print this screen"
    echo "$<script-name> option1 option2 option3"
    exit 1
}

# Process command line arguments
function process-args {
REBUILD_OPT="false"
DEV_OPT="false"
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        help)
        print-help
        shift # past argument
        ;;
        rebuild|REBUILD)
        REBUILD_OPT="true"
        shift # past argument
        ;;
        dev|DEV)
        kube::log::status "NOT IMPLEMENTED YET"
        DEV_OPT="true"
        print-help
        shift # past argument
        ;;
        --default)
        DEFAULT=YES
        ;;
        *)
        kube::log::status "Unknown option provided"
        print-help
                # unknown option
        ;;
    esac
done

}



#
# Main Script execution starts here
#

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../../
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

DIR_BASENAME=$(dirname "${BASH_SOURCE}")
# pushd ${DIR_BASENAME}

kube::log::status "KUBE_ROOT=${KUBE_ROOT} BASH_SOURCE=${BASH_SOURCE} DIR_BASENAME=${DIR_BASENAME} PWD=${PWD} CD=$(cd)"

KUBE_API_SERVER=${KUBE_ROOT}/_output/bin/kube-apiserver
KUBE_SCHEDULER=${KUBE_ROOT}/_output/bin/kube-scheduler
KUBE_CONTROLLER_MANAGER=${KUBE_ROOT}/_output/bin/kube-controller-manager

process-args "$@"

#Remove binaries if needed
remove-binaries

#Remove the temp directories created
kube::log::status "Deleteing tmp dirctory ${PWD}/data/nodes/"
rm -rf ${PWD}/data/nodes/*
kube::log::status "Deleteing tmp dirctory ${PWD}/logs/"
rm -rf ${PWD}/logs/*

# below trap is equivalent to "defer cleanup()" in go
trap cleanup EXIT

kube::etcd::start
apiserver::start
scheduler::start
controllermanager::start

# Run tests for app controllers
kube::log::status "Running integration tests for Kube Apps and related components"

export API_SRV_PID
export SCHED_PID
export CM_PID
export KUBE_ROOT



#Run the actual Command 
TIME="Took %E" time go test -ldflags "-X k8s.io/kubernetes/test/integration/apps.RunBySuite=True" -v -stderrthreshold=FATAL -log_dir=${PWD}/logs/ .

kube::log::status "Integration test for apps completed"