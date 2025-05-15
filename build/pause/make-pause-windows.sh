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
set -x

parent_base_image=$1
kernelversion=$2

# kernelversionImageDigests maps the container images digests for different Windows Server partial os.version from image manifest list
# kernelversionImageDigests["20348"]="sha256:8f49c039657e67cb54861d82acdb907104e268009f1bea4f75b3dec6c6b3d52d"
declare -A kernelversionImageDigests
image_manifest=$(docker manifest inspect "${parent_base_image}")

for images in $(echo "${image_manifest}" | jq '.manifests' | jq -c '.[]'); do
             partial_kernelversion=$(echo "${images}" | jq '.platform."os.version"' | awk -F. '{print $3}')
             image_digest=$(echo "${images}" | jq '.digest' | awk -F\" '{print $2}')
             kernelversionImageDigests[${partial_kernelversion}]=${image_digest}
done

kernelversion_image_digest="${kernelversionImageDigests[${kernelversion}]}"
kernelversion_image_name=$(echo "${parent_base_image}" | awk -F: '{print $1}')@${kernelversion_image_digest}

echo "${kernelversion_image_name}"