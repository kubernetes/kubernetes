# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# Build the etcd image
#
# Usage:
# 	[TAG=2.2.1] [REGISTRY=gcr.io/google_containers] [ARCH=amd64] [BASEIMAGE=busybox] make (build|push)

TAG?=2.2.1
ARCH?=amd64
REGISTRY?=gcr.io/google_containers
TEMP_DIR:=$(shell mktemp -d)

# Default base images for different architectures
# '/' in images has to be written as '\\/'
ifeq ($(ARCH),amd64)
	BASEIMAGE?=busybox
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=hypriot\\/armhf-busybox
endif

build:
	# Download etcd source in a golang container and cross-compile it statically
	# or download official binaries if the ARCH is amd64
	./build-etcd.sh ${TAG} ${ARCH} ${TEMP_DIR}

	# Copy the content in this dir to the temp dir
	cp ./* ${TEMP_DIR}

	# Replace BASEIMAGE with the real base image
	cd ${TEMP_DIR} && sed -i "s/BASEIMAGE/${BASEIMAGE}/g" Dockerfile

	# And build the image
	docker build -t ${REGISTRY}/etcd-${ARCH}:${TAG} ${TEMP_DIR}

ifeq ($(ARCH),amd64)
	# Backward compatibility. TODO: deprecate this image tag
	docker tag -f ${REGISTRY}/etcd-${ARCH}:${TAG} ${REGISTRY}/etcd:${TAG}
endif

push: build
	gcloud docker push ${REGISTRY}/etcd-${ARCH}:${TAG}

	# Backward compatibility. TODO: deprecate this image tag
ifeq ($(ARCH),amd64)
	gcloud docker push ${REGISTRY}/etcd:${TAG}
endif

all: build
.PHONY:	build push