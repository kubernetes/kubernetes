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
#   [ARCH=amd64] [REGISTRY="staging-k8s.gcr.io"] [BIN_DIR="../../../../_output/bin"] make (build|push) VERSION={some_version_number e.g. 0.1}

# SYSTEM_SPEC_NAME is the name of the system spec used for the node conformance
# test. The specs are expected to be in SYSTEM_SPEC_DIR.
SYSTEM_SPEC_NAME?=
SYSTEM_SPEC_DIR?=../../system/specs

# TODO(random-liu): Add this into release progress.
REGISTRY?=staging-k8s.gcr.io
ARCH?=amd64
# BIN_DIR is the directory to find binaries, overwrite with ../../../../_output/bin
# for local development.
BIN_DIR?=../../../../_output/dockerized/bin/linux/${ARCH}
TEMP_DIR:=$(shell mktemp -d)

BASEIMAGE_amd64=debian:jessie
BASEIMAGE_arm=arm32v7/debian:jessie
BASEIMAGE_arm64=arm64v8/debian:jessie
BASEIMAGE_ppc64le=ppc64le/debian:jessie

BASEIMAGE?=${BASEIMAGE_${ARCH}}

IMAGE_NAME:=${REGISTRY}/node-test
COPY_SYSTEM_SPEC_FILE=
SYSTEM_SPEC_FILE_PATH=
ifneq ($(strip $(SYSTEM_SPEC_NAME)),)
    IMAGE_NAME:=${IMAGE_NAME}-${SYSTEM_SPEC_NAME}
    COPY_SYSTEM_SPEC_FILE="'COPY system-spec.yaml /usr/local/etc/'"
    SYSTEM_SPEC_FILE_PATH="'/usr/local/etc/system-spec.yaml'"
endif

all: build

build:

ifndef VERSION
    $(error VERSION is undefined)
endif
	cp -r ./* ${TEMP_DIR}

	cp ${BIN_DIR}/ginkgo ${TEMP_DIR}
	cp ${BIN_DIR}/e2e_node.test ${TEMP_DIR}
ifneq ($(strip $(SYSTEM_SPEC_NAME)),)
	cp ${SYSTEM_SPEC_DIR}/${SYSTEM_SPEC_NAME}.yaml ${TEMP_DIR}/system-spec.yaml
endif

	cd ${TEMP_DIR} && sed -i.back \
	    "s|BASEIMAGE|${BASEIMAGE}|g;\
	     s|COPY_SYSTEM_SPEC_FILE|${COPY_SYSTEM_SPEC_FILE}|g;\
	     s|SYSTEM_SPEC_NAME|${SYSTEM_SPEC_NAME}|g;\
	     s|SYSTEM_SPEC_FILE_PATH|${SYSTEM_SPEC_FILE_PATH}|g" Dockerfile

	# Make scripts executable before they are copied into the Docker image. If we make them executable later, in another layer
	# they'll take up twice the space because the new executable binary differs from the old one, but everything is cached in layers.
	cd ${TEMP_DIR} && chmod a+rx \
		e2e_node.test \
		ginkgo

	docker build --pull -t ${IMAGE_NAME}-${ARCH}:${VERSION} ${TEMP_DIR}

push: build
	docker push ${IMAGE_NAME}-${ARCH}:${VERSION}
ifeq ($(ARCH),amd64)
	docker tag ${IMAGE_NAME}-${ARCH}:${VERSION} ${IMAGE_NAME}:${VERSION}
	docker push ${IMAGE_NAME}:${VERSION}
endif

.PHONY: all
