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

# Script to validate the web ui e2e protractor tests work as expected.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

if [[ "${SETUP_SHIPPABLE}" == "true" ]]; then
  XVFB=/usr/bin/Xvfb; XVFBARGS=":1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset"; PIDFILE=/var/run/xvfb.pid;
  export DISPLAY=:1
  start-stop-daemon --start --quiet --pidfile $PIDFILE --make-pidfile --background --exec $XVFB -- $XVFBARGS
  sleep 3 # give xvfb some time to start
  nohup bash -c "./node_modules/protractor/bin/webdriver-manager start 2>&1 &"
  sleep 3 # give xvfb some time to start
fi

# The api version in which objects are currently stored in etcd.
KUBE_OLD_API_VERSION=${KUBE_OLD_API_VERSION:-"v1beta1"}
# The api version in which our etcd objects should be converted to.
# The new api version
KUBE_NEW_API_VERSION=${KUBE_NEW_API_VERSION:-"v1beta3"}

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-4001}
API_PORT=${API_PORT:-8080}
KUBELET_PORT=${KUBELET_PORT:-10250}
KUBE_API_VERSIONS=""
RUNTIME_CONFIG=""
BROWSER_NAME=${BROWSER_NAME:-firefox}

KUBECTL="${KUBE_OUTPUT_HOSTBIN}/kubectl"

trap kube::apiserver::cleanup EXIT SIGINT

which protractor >/dev/null || {
  kube::log::usage "protractor must be in your PATH"
  exit 1
}

kube::etcd::start

kube::log::status "Running tests for web ui e2e scenario"

#######################################################
# Start a server which supports only the new api version.
#######################################################

KUBE_API_VERSIONS="${KUBE_NEW_API_VERSION}"
RUNTIME_CONFIG="api/all=false,api/${KUBE_NEW_API_VERSION}=true"

kube::apiserver::start

kube::log::status "Creating a pod"
${KUBECTL} create -f examples/pod.yaml

kube::log::status "Running e2e tests"

protractor "${KUBE_ROOT}/www/master/protractor/conf.js" --baseUrl http://localhost:8080/static/app/ --capabilities.browserName "${BROWSER_NAME}"
