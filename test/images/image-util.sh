#!/bin/bash

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

TASK=$1
IMAGE=$2

# Mapping of go ARCH to actual architectures shipped part of multiarch/qemu-user-static project
declare -A QEMUARCHS=( ["amd64"]="x86_64" ["arm"]="arm" ["arm64"]="aarch64" ["ppc64le"]="ppc64le" ["s390x"]="s390x" )

# Returns list of all supported architectures from BASEIMAGE file
listArchs() {
  cut -d "=" -f 1 ${IMAGE}/BASEIMAGE
}

# Returns baseimage need to used in Dockerfile for any given architecture
getBaseImage() {
  arch=$1
  echo $(grep "${arch}=" BASEIMAGE | cut -d= -f2)
}

# This function will build test image for all the architectures
# mentioned in BASEIMAGE file. In the absence of BASEIMAGE file,
# it will build for all the supported arch list - amd64, arm,
# arm64, ppc64le, s390x
build() {
  if [[ -f ${IMAGE}/BASEIMAGE ]]; then
    archs=$(listArchs)
  else
    archs=${!QEMUARCHS[@]}
  fi

  for arch in ${archs}; do
    echo "Building image for ${IMAGE} ARCH: ${arch}..."

    # Create a temporary directory for every architecture and copy the image content
    # and build the image from temporary directory
    temp_dir=$(mktemp -d)
    cp -r ${IMAGE}/* ${temp_dir}
    if [[ -f ${IMAGE}/Makefile ]]; then
      # make bin will take care of all the prerequisites needed
      # for building the docker image
      make -C ${IMAGE} bin ARCH=${arch} TARGET=${temp_dir}
    fi
    pushd ${temp_dir}
    # image tag
    TAG=$(<VERSION)
    # image name
    IMAGENAME=$(<NAME)

    if [[ -f BASEIMAGE ]]; then
      BASEIMAGE=$(getBaseImage ${arch})
      sed -i "s|BASEIMAGE|${BASEIMAGE}|g" Dockerfile
    fi

    # copy the qemu-*-static binary to docker image to build the multi architecture image on x86 platform
    if [[ $(grep "CROSS_BUILD_" Dockerfile) ]]; then
      if [[ "${arch}" == "amd64" ]]; then
        sed -i "/CROSS_BUILD_/d" Dockerfile
      else
        sed -i "s|QEMUARCH|${QEMUARCHS[$arch]}|g" Dockerfile
        # Register qemu-*-static for all supported processors except the current one
        docker run --rm --privileged multiarch/qemu-user-static:register --reset
        curl -sSL https://github.com/multiarch/qemu-user-static/releases/download/${QEMUVERSION}/x86_64_qemu-${QEMUARCHS[$arch]}-static.tar.gz | tar -xz -C ${temp_dir}
        sed -i "s/CROSS_BUILD_//g" Dockerfile
      fi
    fi

    docker build --pull -t ${REGISTRY}/${IMAGENAME}-${arch}:${TAG} .

    # Image without any arch postfix will point to amd64 by default.
    if [[ "${arch}" == "amd64" ]]; then
      docker tag ${REGISTRY}/${IMAGENAME}-${arch}:${TAG} ${REGISTRY}/${IMAGENAME}:${TAG}
    fi
    popd
  done
}

# This function will push the docker images
push() {
  if [[ -f ${IMAGE}/BASEIMAGE ]]; then
    archs=$(listArchs)
  else
    archs=${!QEMUARCHS[@]}
  fi
  for arch in ${archs}; do
    IMAGENAME=$(<${IMAGE}/NAME)
    TAG=$(<${IMAGE}/VERSION)
    gcloud docker -- push ${REGISTRY}/${IMAGENAME}-${arch}:${TAG}
    if [[ "${arch}" == "amd64" ]]; then
      gcloud docker -- push ${REGISTRY}/${IMAGENAME}:${TAG}
    fi
  done
}

eval ${TASK}
