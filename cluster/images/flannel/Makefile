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

# Build the flannel image
#
# Usage:
# 	[TAG=0.5.5] [REGISTRY=gcr.io/google_containers] [ARCH=amd64] make build

TAG?=0.5.5
ARCH?=amd64
REGISTRY?=gcr.io/google_containers
KUBE_CROSS_TAG=$(shell cat ../../../build-image/cross/VERSION)
GOARM=6
TEMP_DIR:=$(shell mktemp -d)
BASEIMAGE?=gcr.io/google_containers/debian-iptables-${ARCH}:v2

ifeq ($(ARCH),arm)
	CC=arm-linux-gnueabi-gcc
endif
ifeq ($(ARCH),arm64)
	CC=aarch64-linux-gnu-gcc
endif
ifeq ($(ARCH),ppc64le)
	CC=powerpc64le-linux-gnu-gcc
endif

build:
ifeq ($(ARCH),amd64)
	# If we should build an amd64 flannel image, go with the official one
	docker pull quay.io/coreos/flannel:$(TAG)

	docker tag -f quay.io/coreos/flannel:$(TAG) $(REGISTRY)/flannel-$(ARCH):$(TAG)
else
	# Copy the content in this dir to the temp dir
	cp ./* $(TEMP_DIR)

	docker run -it -v $(TEMP_DIR):/flannel/bin gcr.io/google_containers/kube-cross:$(KUBE_CROSS_TAG) /bin/bash -c \
		"curl -sSL https://github.com/coreos/flannel/archive/v${TAG}.tar.gz | tar -C /flannel -xz --strip-components=1 \
		&& cd /flannel && GOARM=$(GOARM) GOARCH=$(ARCH) CC=$(CC) CGO_ENABLED=1 ./build"

	# Replace BASEIMAGE with the real base image
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

	# Download mk-docker-opts.sh
	curl -sSL https://raw.githubusercontent.com/coreos/flannel/v$(TAG)/dist/mk-docker-opts.sh > $(TEMP_DIR)/mk-docker-opts.sh

	# And build the image
	docker build -t $(REGISTRY)/flannel-$(ARCH):$(TAG) $(TEMP_DIR)
endif

push: build
	gcloud docker push $(REGISTRY)/flannel-$(ARCH):$(TAG)

all: build
.PHONY: build push
