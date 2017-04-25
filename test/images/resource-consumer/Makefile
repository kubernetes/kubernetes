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

TAG ?= beta5
REGISTRY = gcr.io/google_containers
ARCH ?= $(shell go env GOARCH)
ALL_ARCH = amd64 arm arm64 ppc64le

CONTRIB_RELEASE = fix_import_path
GOLANG_VERSION=1.7.5
GOARM=7
QEMUVERSION=v2.7.0

IMGNAME = resource_consumer
IMAGE = $(REGISTRY)/$(IMGNAME)
MULTI_ARCH_IMG = $(IMAGE)-$(ARCH)

CTRL_IMGNAME = resource_consumer/controller
CTRL_IMAGE = $(REGISTRY)/$(CTRL_IMGNAME)
CTRL_MULTI_ARCH_IMG = $(CTRL_IMAGE)-$(ARCH)

# Set default base image dynamically for each arch
ifeq ($(ARCH),amd64)
	BASEIMAGE?=ubuntu:16.04
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=armhf/ubuntu:16.04
	QEMUARCH=arm
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=aarch64/ubuntu:16.04
	QEMUARCH=aarch64
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/ubuntu:16.04
	QEMUARCH=ppc64le
endif

TEMP_DIR := $(shell mktemp -d)

# If you want to build AND push all containers, see the 'all-push' rule.
all: all-container

sub-container-%:
	$(MAKE) ARCH=$* container

sub-push-%:
	$(MAKE) ARCH=$* push

all-container: $(addprefix sub-container-,$(ALL_ARCH))

all-push: $(addprefix sub-push-,$(ALL_ARCH))

container: .container-$(ARCH)
.container-$(ARCH):
	cp -r ./* $(TEMP_DIR)
	# Set the base image
	cd $(TEMP_DIR) && sed -i.bak 's|BASEIMAGE|$(BASEIMAGE)|g' Dockerfile
	cd $(TEMP_DIR) && sed -i "s|ARCH|$(QEMUARCH)|g" Dockerfile

	docker run -it -v $(TEMP_DIR):/build:Z -v `pwd`/../../../:/go/src/k8s.io/kubernetes:Z \
	golang:$(GOLANG_VERSION) \
	/bin/bash -c "\
            cd /go/src/k8s.io/kubernetes/test/images/resource-consumer && \
            CGO_ENABLED=0 GOARM=$(GOARM) GOARCH=$(ARCH) go build -a -installsuffix cgo --ldflags '-w' -o /build/consume-cpu/consume-cpu ./consume-cpu/consume_cpu.go && \
            CGO_ENABLED=0 GOARM=$(GOARM) GOARCH=$(ARCH) go build -a -installsuffix cgo --ldflags '-w' -o /build/consumer . && \
            CGO_ENABLED=0 GOARM=$(GOARM) GOARCH=$(ARCH) go build -a -installsuffix cgo --ldflags '-w' -o /build/controller/controller ./controller/controller.go"

ifeq ($(ARCH),amd64)
	# When building "normally" for amd64, remove the whole line, it has no part in the amd64 image
	cd $(TEMP_DIR) && sed -i "/CROSS_BUILD_/d" Dockerfile
else
	# When cross-building, only the placeholder "CROSS_BUILD_" should be removed
	# Register /usr/bin/qemu-ARCH-static as the handler for ARM binaries in the kernel
	docker run --rm --privileged multiarch/qemu-user-static:register --reset
	curl -sSL https://github.com/multiarch/qemu-user-static/releases/download/$(QEMUVERSION)/x86_64_qemu-$(QEMUARCH)-static.tar.gz | tar -xz -C $(TEMP_DIR)
	cd $(TEMP_DIR) && sed -i "s/CROSS_BUILD_//g" Dockerfile
endif
	docker build --pull -t $(MULTI_ARCH_IMG):$(TAG) $(TEMP_DIR)
	docker build --pull -t $(CTRL_MULTI_ARCH_IMG):$(TAG) $(TEMP_DIR)/controller


ifeq ($(ARCH), amd64)
	# This is for to maintain the backward compatibility
	docker tag $(MULTI_ARCH_IMG):$(TAG) $(IMAGE):$(TAG)
	docker tag $(CTRL_MULTI_ARCH_IMG):$(TAG) $(CTRL_IMAGE):$(TAG)
endif

push: .push-$(ARCH)
.push-$(ARCH): .container-$(ARCH)
	gcloud docker -- push $(MULTI_ARCH_IMG):$(TAG)
	gcloud docker -- push $(CTRL_MULTI_ARCH_IMG):$(TAG)

ifeq ($(ARCH),amd64)
	gcloud docker -- push $(IMAGE):$(TAG)
	gcloud docker -- push $(CTRL_IMAGE):$(TAG)
endif

clean:
