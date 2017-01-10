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

.PHONY:	build push

REGISTRY?="gcr.io/google_containers"
IMAGE=debian-iptables
TAG=v4
ARCH?=amd64
TEMP_DIR:=$(shell mktemp -d)

ifeq ($(ARCH),amd64)
	BASEIMAGE?=debian:jessie
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=armel/debian:jessie
	QEMUARCH=arm
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=aarch64/debian:jessie
	QEMUARCH=aarch64
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/debian:jessie
	QEMUARCH=ppc64le
endif

build:
	cp ./* $(TEMP_DIR)
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile
	cd $(TEMP_DIR) && sed -i "s|ARCH|$(QEMUARCH)|g" Dockerfile

ifeq ($(ARCH),amd64)
	# When building "normally" for amd64, remove the whole line, it has no part in the amd64 image
	cd $(TEMP_DIR) && sed -i "/CROSS_BUILD_/d" Dockerfile
else
	# When cross-building, only the placeholder "CROSS_BUILD_" should be removed
	# Register /usr/bin/qemu-ARCH-static as the handler for ARM binaries in the kernel
	docker run --rm --privileged multiarch/qemu-user-static:register --reset
	curl -sSL https://github.com/multiarch/qemu-user-static/releases/download/v2.5.0/x86_64_qemu-$(QEMUARCH)-static.tar.xz | tar -xJ -C $(TEMP_DIR)
	cd $(TEMP_DIR) && sed -i "s/CROSS_BUILD_//g" Dockerfile
endif

	docker build --pull -t $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG) $(TEMP_DIR)

push: build
	gcloud docker -- push $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG)

all: push
