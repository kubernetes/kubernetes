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

all: build

REGISTRY ?= gcr.io/google-containers
IMAGE ?= debian-base
BUILD_IMAGE ?= debian-build

TAG ?= 0.2

TAR_FILE ?= rootfs.tar
ARCH?=amd64
TEMP_DIR:=$(shell mktemp -d)
QEMUVERSION=v2.9.1

ifeq ($(ARCH),amd64)
	BASEIMAGE?=debian:jessie
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=arm32v7/debian:jessie
	QEMUARCH=arm
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=arm64v8/debian:jessie
	QEMUARCH=aarch64
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/debian:jessie
	QEMUARCH=ppc64le
endif
ifeq ($(ARCH),s390x)
	BASEIMAGE?=s390x/debian:jessie
	QEMUARCH=s390x
endif

build: clean
	cp ./* $(TEMP_DIR)
	cat Dockerfile.build \
		| sed "s|BASEIMAGE|$(BASEIMAGE)|g" \
		| sed "s|ARCH|$(QEMUARCH)|g" \
		> $(TEMP_DIR)/Dockerfile.build

ifeq ($(ARCH),amd64)
	# When building "normally" for amd64, remove the whole line, it has no part in the amd64 image
	sed "/CROSS_BUILD_/d" $(TEMP_DIR)/Dockerfile.build > $(TEMP_DIR)/Dockerfile.build.tmp
else
	# When cross-building, only the placeholder "CROSS_BUILD_" should be removed
	# Register /usr/bin/qemu-ARCH-static as the handler for ARM binaries in the kernel
	docker run --rm --privileged multiarch/qemu-user-static:register --reset
	curl -sSL https://github.com/multiarch/qemu-user-static/releases/download/$(QEMUVERSION)/x86_64_qemu-$(QEMUARCH)-static.tar.gz | tar -xz -C $(TEMP_DIR)
	sed "s/CROSS_BUILD_//g" $(TEMP_DIR)/Dockerfile.build > $(TEMP_DIR)/Dockerfile.build.tmp
endif
	mv $(TEMP_DIR)/Dockerfile.build.tmp $(TEMP_DIR)/Dockerfile.build

	docker build --pull -t $(BUILD_IMAGE) -f $(TEMP_DIR)/Dockerfile.build $(TEMP_DIR)
	docker create --name $(BUILD_IMAGE) $(BUILD_IMAGE)
	docker export $(BUILD_IMAGE) > $(TEMP_DIR)/$(TAR_FILE)
	docker build -t $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG) $(TEMP_DIR)
	rm -rf $(TEMP_DIR)

push: build
	gcloud docker -- push $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG)

clean:
	docker rmi -f $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG) || true
	docker rmi -f $(BUILD_IMAGE)   || true
	docker rm  -f $(BUILD_IMAGE)   || true
