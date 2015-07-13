#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# Script to test cluster/update-storage-objects.sh works as expected.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# The api version in which objects are currently stored in etcd.
KUBE_OLD_API_VERSION=${KUBE_OLD_API_VERSION:-"v1beta3"}
# The api version in which our etcd objects should be converted to.
# The new api version
KUBE_NEW_API_VERSION=${KUBE_NEW_API_VERSION:-"v1"}

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-4001}
API_PORT=${API_PORT:-8080}
KUBELET_PORT=${KUBELET_PORT:-10250}
KUBE_API_VERSIONS=""
RUNTIME_CONFIG=""

KUBECTL="${KUBE_OUTPUT_HOSTBIN}/kubectl"
UPDATE_ETCD_OBJECTS_SCRIPT="${KUBE_ROOT}/cluster/update-storage-objects.sh"

trap kube::apiserver::cleanup EXIT SIGINT
    --service-cluster-ip-range="10.0.0.0/24" 1>&2 &

kube::etcd::start

kube::log::status "Running test for update etcd object scenario"

"${KUBE_ROOT}/hack/build-go.sh"


#######################################################
# Step 1: Start a server which supports both the old and new api versions,
# but KUBE_OLD_API_VERSION is the latest (storage) version.
#######################################################

KUBE_API_VERSIONS="${KUBE_OLD_API_VERSION},${KUBE_NEW_API_VERSION}"
RUNTIME_CONFIG="api/all=false,api/${KUBE_OLD_API_VERSION}=true,api/${KUBE_NEW_API_VERSION}=true"
kube::apiserver::start

# Create a pod
kube::log::status "Creating a pod"
${KUBECTL} create -f examples/pod.yaml

kube::apiserver::kill


#######################################################
# Step 2: Start a server which supports both the old and new api versions,
# but KUBE_NEW_API_VERSION is the latest (storage) version.
#######################################################

KUBE_API_VERSIONS="${KUBE_NEW_API_VERSION},${KUBE_OLD_API_VERSION}"
RUNTIME_CONFIG="api/all=false,api/${KUBE_OLD_API_VERSION}=true,api/${KUBE_NEW_API_VERSION}=true"
kube::apiserver::start

# Update etcd objects, so that will now be stored in the new api version.
${UPDATE_ETCD_OBJECTS_SCRIPT}

kube::apiserver::kill


#######################################################
# Step 3 : Start a server which supports only the new api version.
#######################################################

KUBE_API_VERSIONS="${KUBE_NEW_API_VERSION}"
RUNTIME_CONFIG="api/all=false,api/${KUBE_NEW_API_VERSION}=true"
kube::apiserver::start

# Verify that the server is able to read the object.
# This will fail if the object is in a version that is not understood by the
# master.
${KUBECTL} get pods
