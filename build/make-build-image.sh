#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

# Build the docker image necessary for building Kubernetes
#
# This script will package the parts of the repo that we need to build
# Kubernetes into a tar file and put it in the right place in the output
# directory.  It will then copy over the Dockerfile and build the kube-build
# image.
set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT="$(dirname "${BASH_SOURCE}")/.."
source "${KUBE_ROOT}/build/common.sh"

kube::build::verify_prereqs
kube::build::build_image
