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

# Generates secret, creates secret on kube, creates pod on kube

set -o errexit
set -o nounset
set -o pipefail

ORIGIN=$(dirname "${BASH_SOURCE}")
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

## Generate resources
${ORIGIN}/resource-generator.sh

## Create the secret
${KUBE_ROOT}/cluster/kubectl.sh create -f ${ORIGIN}/secret.json

## Create the pod
${KUBE_ROOT}/cluster/kubectl.sh create -f ${ORIGIN}/pod.json

## Create the services
${KUBE_ROOT}/cluster/kubectl.sh create -f ${ORIGIN}/api-service.json
${KUBE_ROOT}/cluster/kubectl.sh create -f ${ORIGIN}/ui-service.json