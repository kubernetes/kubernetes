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

# Cross-build the mount-tester image
#
# Usage:
#       [TAG=v1.0] [REGISTRY=gcr.io/google_containers] [ARCH=amd64] [BASEIMAGE=busybox] make all

.PHONY: all push container clean

TAG ?= 0.1

REGISTRY ?= gcr.io/google_containers

# Architectures supported: amd64, arm, arm64, ppc64le and s390x
ARCH ?= amd64

ALL_ARCH = amd64 arm arm64 ppc64le s390x

GOARM=7
TEMP_DIR := $(shell mktemp -d)
GOLANG_VERSION=1.7.5

BIN = n-way-http
SRCS = server.go

IMAGE = $(REGISTRY)/$(BIN)
MULTI_ARCH_IMG = $(IMAGE)-$(ARCH)

# Set default base image dynamically for each arch
ifeq ($(ARCH),amd64)
        BASEIMAGE?=busybox
endif
ifeq ($(ARCH),arm)
        BASEIMAGE?=armhf/busybox
endif
ifeq ($(ARCH),arm64)
        BASEIMAGE?=aarch64/busybox
endif
ifeq ($(ARCH),ppc64le)
        BASEIMAGE?=ppc64le/busybox
endif
ifeq ($(ARCH),s390x)
        BASEIMAGE?=s390x/busybox
endif

# If you want to build AND push all containers, see the 'all-push' rule.
all: all-container

sub-container-%:
	$(MAKE) ARCH=$* container

sub-push-%:
	$(MAKE) ARCH=$* push

all-container: $(addprefix sub-container-,$(ALL_ARCH))

all-push: $(addprefix sub-push-,$(ALL_ARCH))

build: bin/$(BIN)-$(ARCH)

bin/$(BIN)-$(ARCH): $(SRCS)
	# Copy the content in this dir to the temp dir
	cp ./* $(TEMP_DIR)

	docker run -it -v $(TEMP_DIR):/build \
	golang:$(GOLANG_VERSION) \
	/bin/bash -c "\
            cd /build && \
            CGO_ENABLED=0 GOARM=$(GOARM) GOARCH=$(ARCH) go build -a -installsuffix cgo --ldflags '-w' -o $(BIN) ./$(SRCS)"

container: .container-$(ARCH)
.container-$(ARCH): bin/$(BIN)-$(ARCH)
	# Set the base image
	cd $(TEMP_DIR) && sed -i.bak 's|BASEIMAGE|$(BASEIMAGE)|g' Dockerfile

	docker build --pull -t $(IMAGE):$(TAG) $(TEMP_DIR)

ifeq ($(ARCH), amd64)
	# This is for to maintain the backward compatibility
	docker tag $(MULTI_ARCH_IMG):$(TAG) $(IMAGE):$(TAG)
endif

push: .push-$(ARCH)
.push-$(ARCH): .container-$(ARCH)
	gcloud docker -- push $(IMAGE):$(TAG)

ifeq ($(ARCH),amd64)
	gcloud docker -- push $(IMAGE):$(TAG)
endif

clean:
	rm -rf $(BIN)