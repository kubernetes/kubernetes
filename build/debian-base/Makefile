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

all: all-build

REGISTRY ?= staging-k8s.gcr.io
IMAGE ?= $(REGISTRY)/debian-base

TAG ?= v2.0.0

ARCH?=amd64
ALL_ARCH = amd64 arm arm64 ppc64le s390x

TEMP_DIR:=$(shell mktemp -d)
QEMUVERSION=v2.9.1

# This option is for running docker manifest command
export DOCKER_CLI_EXPERIMENTAL := enabled

BUILD_BASE=buildBaseWithQemu

ifeq ($(ARCH),amd64)
	BASEIMAGE?=debian:buster-slim
	BUILD_BASE=$(BASEIMAGE)
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=arm32v7/debian:buster-slim
	QEMUARCH=arm
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=arm64v8/debian:buster-slim
	QEMUARCH=aarch64
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/debian:buster-slim
	QEMUARCH=ppc64le
endif
ifeq ($(ARCH),s390x)
	BASEIMAGE?=s390x/debian:buster-slim
	QEMUARCH=s390x
endif

QEMU_URL=https://github.com/multiarch/qemu-user-static/releases/download/$(QEMUVERSION)/x86_64_qemu-$(QEMUARCH)-static.tar.gz

sub-build-%:
	$(MAKE) ARCH=$* build

all-build: $(addprefix sub-build-,$(ALL_ARCH))

sub-push-image-%:
	$(MAKE) ARCH=$* push

all-push-images: $(addprefix sub-push-image-,$(ALL_ARCH))

all-push: all-push-images push-manifest

push-manifest:
	docker manifest create --amend $(IMAGE):$(TAG) $(shell echo $(ALL_ARCH) | sed -e "s~[^ ]*~$(IMAGE)\-&:$(TAG)~g")
	@for arch in $(ALL_ARCH); do docker manifest annotate --arch $${arch} ${IMAGE}:${TAG} ${IMAGE}-$${arch}:${TAG}; done
	docker manifest push --purge ${IMAGE}:${TAG}

build: clean
	@echo '## Building $(ARCH) image'

	@cp ./* $(TEMP_DIR)

ifneq ($(ARCH),amd64)
	@# Register /usr/bin/qemu-ARCH-static as the handler for non-x86 binaries in the kernel
	../../third_party/multiarch/qemu-user-static/register/register.sh --reset
	@# The Dockerfile expects to find the qemu emulator binary in a 'qemu.tar.gz' in the build context
	curl -sSL $(QEMU_URL) > $(TEMP_DIR)/qemu.tar.gz
endif

	docker build --pull \
		--build-arg BASEIMAGE=$(BASEIMAGE) --build-arg BUILD_BASE=$(BUILD_BASE) \
		--tag $(IMAGE)-$(ARCH):$(TAG) \
		--target main $(TEMP_DIR)

	@rm -rf $(TEMP_DIR)

push: build
	docker push $(IMAGE)-$(ARCH):$(TAG)

clean:
	2>/dev/null docker image rm -f $(IMAGE)-$(ARCH):$(TAG) || true
