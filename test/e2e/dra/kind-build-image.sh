#!/usr/bin/env bash

# Copyright 2022 The Kubernetes Authors.
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

# This scripts invokes `kind build image` so that the resulting
# image has a containerd with CDI support.
#
# Usage: kind-build-image.sh <tag of generated image>

set -ex
set -o pipefail

tag="$1"

# Created manually in the kind repo by bentheelder with
# make -C images/base push EXTRA_BUILD_OPT=--build-arg=CONTAINERD_VERSION=v1.7.1 TAG=$(date +v%Y%m%d)-$(git describe --always --dirty)-containerd_v1.7.1
base_image="gcr.io/k8s-staging-kind/base:v20230515-01914134-containerd_v1.7.1@sha256:468fc430a6848884b786c5cd2f1c03e7a0977f04fb129a2cda2a19ec986ddacb"

kind build node-image --base-image "$base_image"  --image "$tag" "$(pwd)"
