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

# Build the hyperkube base image. This image is used to build the hyperkube image.
#
# Usage:
#   [ARCH=amd64] [REGISTRY="gcr.io/google-containers"] make (build|push)

REGISTRY?=gcr.io/google-containers
IMAGE?=debian-hyperkube-base
TAG=0.6
ARCH?=amd64
CACHEBUST?=1

BASEIMAGE=gcr.io/google-containers/debian-base-$(ARCH):0.2
CNI_VERSION=v0.6.0

TEMP_DIR:=$(shell mktemp -d)
CNI_TARBALL=cni-plugins-$(ARCH)-$(CNI_VERSION).tgz

.PHONY: all build push clean

all: push

cni-tars/$(CNI_TARBALL):
	mkdir -p cni-tars/
	cd cni-tars/ && curl -sSLO --retry 5 https://storage.googleapis.com/kubernetes-release/network-plugins/${CNI_TARBALL}

clean:
	rm -rf cni-tars/

build: cni-tars/$(CNI_TARBALL)
	cp Dockerfile $(TEMP_DIR)
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

ifeq ($(CACHEBUST),1)
	cd ${TEMP_DIR} && sed -i.back "s|CACHEBUST|$(shell uuidgen)|g" Dockerfile
endif

	mkdir -p ${TEMP_DIR}/cni-bin/bin
	tar -xz -C ${TEMP_DIR}/cni-bin/bin -f "cni-tars/${CNI_TARBALL}"

	# Register /usr/bin/qemu-ARCH-static as the handler for non-x86 binaries in the kernel
	docker run --rm --privileged multiarch/qemu-user-static:register --reset
	docker build --pull -t $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG) $(TEMP_DIR)
	rm -rf $(TEMP_DIR)

push: build
	gcloud docker -- push $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG)
