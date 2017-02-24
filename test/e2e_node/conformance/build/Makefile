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

# Build the node-test image.
#
# Usage:
#   [ARCH=amd64] [REGISTRY="gcr.io/google_containers"] [BIN_DIR="../../../../_output/bin"] make (build|push) VERSION={some_version_number e.g. 0.1}

# TODO(random-liu): Add this into release progress.
REGISTRY?=gcr.io/google_containers
ARCH?=amd64
# BIN_DIR is the directory to find binaries, overwrite with ../../../../_output/bin
# for local development.
BIN_DIR?=../../../../_output/dockerized/bin/linux/${ARCH}
TEMP_DIR:=$(shell mktemp -d)

BASEIMAGE_amd64=debian:jessie
BASEIMAGE_arm=armhf/debian:jessie
BASEIMAGE_arm64=aarch64/debian:jessie
BASEIMAGE_ppc64le=ppc64le/debian:jessie

BASEIMAGE?=${BASEIMAGE_${ARCH}}

all: build

build:

ifndef VERSION
    $(error VERSION is undefined)
endif
	cp -r ./* ${TEMP_DIR}

	cp ${BIN_DIR}/ginkgo ${TEMP_DIR}
	cp ${BIN_DIR}/e2e_node.test ${TEMP_DIR}

	cd ${TEMP_DIR} && sed -i.back "s|BASEIMAGE|${BASEIMAGE}|g" Dockerfile

	# Make scripts executable before they are copied into the Docker image. If we make them executable later, in another layer
	# they'll take up twice the space because the new executable binary differs from the old one, but everything is cached in layers.
	cd ${TEMP_DIR} && chmod a+rx \
		e2e_node.test \
		ginkgo

	docker build --pull -t ${REGISTRY}/node-test-${ARCH}:${VERSION} ${TEMP_DIR}

push: build
	gcloud docker push ${REGISTRY}/node-test-${ARCH}:${VERSION}
ifeq ($(ARCH),amd64)
	docker tag ${REGISTRY}/node-test-${ARCH}:${VERSION} ${REGISTRY}/node-test:${VERSION}
	gcloud docker -- push ${REGISTRY}/node-test:${VERSION}
endif

.PHONY: all
