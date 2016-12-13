#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

# Build a Kubernetes release.  This will build the binaries, create the Docker
# images and other build artifacts.
#
# For pushing these artifacts publicly to Google Cloud Storage or to a registry
# please refer to the kubernetes/release repo at
# https://github.com/kubernetes/release.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

minikube start --kubernetes-version="file://${KUBE_ROOT}/_output/dockerized/bin/linux/amd64/localkube"