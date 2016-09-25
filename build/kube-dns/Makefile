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

# Makefile for the Docker image gcr.io/google_containers/kubedns-<ARCH>
# MAINTAINER: Tim Hockin <thockin@google.com>
# If you update this image please bump the tag value before pushing.
#
# Usage:
#   [ARCH=amd64] [TAG=1.6] [REGISTRY=gcr.io/google_containers] [BASEIMAGE=busybox] make (container|push)

# Default registry, arch and tag. This can be overwritten by arguments to make
PLATFORM?=linux
ARCH?=amd64
TAG?=1.7
REGISTRY?=gcr.io/google_containers

GOLANG_VERSION=1.6
KUBE_ROOT=$(shell pwd)/../..
TEMP_DIR:=$(shell mktemp -d)

ifeq ($(ARCH),amd64)
	BASEIMAGE?=busybox
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=armel/busybox
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=aarch64/busybox
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/busybox
endif


all: container

container:
	# Copy the content in this dir to the temp dir
	cp $(KUBE_ROOT)/_output/dockerized/bin/$(PLATFORM)/$(ARCH)/kube-dns $(TEMP_DIR)
	cp $(KUBE_ROOT)/build/kube-dns/Dockerfile $(TEMP_DIR)

	# Replace BASEIMAGE with the real base image
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

	# And build the image
	docker build -t $(REGISTRY)/kubedns-$(ARCH):$(TAG) $(TEMP_DIR)

	# delete temp dir
	rm -rf $(TEMP_DIR)

push: container
	gcloud docker push $(REGISTRY)/kubedns-$(ARCH):$(TAG)

.PHONY: all container push
