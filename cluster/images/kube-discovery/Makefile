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

# Build the kube-discovery image.
#
# Requires a pre-built kube-discovery binary:
#   build/run.sh /bin/bash -c "KUBE_BUILD_PLATFORMS=linux/ARCH make WHAT=cmd/kube-discovery"
#
# Usage:
#   [ARCH=amd64] [REGISTRY="gcr.io/google_containers"] make (build|push) VERSION={some_released_version_of_kubernetes}

REGISTRY?=gcr.io/google_containers
ARCH?=amd64
TEMP_DIR:=$(shell mktemp -d)
VERSION?=1.0

ifeq ($(ARCH),amd64)
	BASEIMAGE?=debian:jessie
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=armel/debian:jessie
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=aarch64/debian:jessie
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/debian:jessie
endif
ifeq ($(ARCH),s390x)
	BASEIMAGE?=s390x/debian:jessie
endif


all: build

build:
	cp -r ./* ${TEMP_DIR}
	cp ../../../_output/dockerized/bin/linux/${ARCH}/kube-discovery ${TEMP_DIR}
	cd ${TEMP_DIR} && sed -i.back "s|BASEIMAGE|${BASEIMAGE}|g" Dockerfile
	docker build --pull -t ${REGISTRY}/kube-discovery-${ARCH}:${VERSION} ${TEMP_DIR}
	rm -rf "${TEMP_DIR}"

push: build
	gcloud docker -- push ${REGISTRY}/kube-discovery-${ARCH}:${VERSION}

.PHONY: all
