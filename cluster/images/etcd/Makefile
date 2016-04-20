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

TAG?=2.2.5
ARCH?=amd64
REGISTRY?=gcr.io/google_containers
GOLANG_VERSION?=1.6.0
GOARM=6
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

build:
	# Copy the content in this dir to the temp dir
	cp ./* $(TEMP_DIR)

ifeq ($(ARCH),amd64)

	# Do not compile if we should make an image for amd64, use the official etcd binaries instead
	curl -sSL --retry 5 https://github.com/coreos/etcd/releases/download/v$(TAG)/etcd-v$(TAG)-linux-amd64.tar.gz | tar -xz -C $(TEMP_DIR) --strip-components=1
else
	
	# Download etcd in a golang container and cross-compile it statically
	docker run -it -v $(TEMP_DIR):/etcdbin golang:$(GOLANG_VERSION) /bin/bash -c \
		"git clone https://github.com/coreos/etcd \
		&& cd etcd && git checkout v$(TAG) \
		&& GOARM=$(GOARM) GOARCH=$(ARCH) ./build \
		&& cp bin/* /etcdbin"
endif

	# Replace BASEIMAGE with the real base image
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

	# And build the image
	docker build -t $(REGISTRY)/etcd-$(ARCH):$(TAG) $(TEMP_DIR)

push: build
	gcloud docker push $(REGISTRY)/etcd-$(ARCH):$(TAG)

ifeq ($(ARCH),amd64)
	# Backward compatibility. TODO: deprecate this image tag
	docker tag -f $(REGISTRY)/etcd-$(ARCH):$(TAG) $(REGISTRY)/etcd:$(TAG)
	gcloud docker push $(REGISTRY)/etcd:$(TAG)
endif

all: build
.PHONY:	build push