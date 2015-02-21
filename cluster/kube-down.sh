#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# Tear down a Kubernetes cluster.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

# Delete services which use external load balancer.
function teardown-external-services() {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local template='{{ range .items }}{{ .id }}{{ " " }}{{ .createExternalLoadBalancer }}{{ "\n" }}{{ end }}'
  local oldifs="${IFS}"
  IFS=$'\n'
  for service in $("${kubectl}" get se -o template --template="${template}"); do
    local id=$(echo $service | cut -f 1 -d " ")
    local external=$(echo "${service}" | cut -f 2 -d " ")
    if [ "${external}" = "true" ]
    then
      "${kubectl}" delete service "${id}"
    fi
  done
  IFS="${oldifs}"
}

echo "Bringing down cluster using provider: $KUBERNETES_PROVIDER"

verify-prereqs
teardown-monitoring-firewall
teardown-logging-firewall
teardown-external-services

kube-down

echo "Done"
