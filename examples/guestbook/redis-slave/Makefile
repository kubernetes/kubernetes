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

.PHONY: all push container clean

TAG ?= v2

REGISTRY ?= gcr.io/google-containers

# Architectures supported: amd64, arm, arm64, ppc64le
ARCH ?= $(shell go env GOARCH)

ALL_ARCH = amd64 arm arm64 ppc64le

TEMP_DIR := $(shell mktemp -d)

IMGNAME = gb-redisslave
IMAGE = $(REGISTRY)/$(IMGNAME)
MULTI_ARCH_IMG = $(IMAGE)-$(ARCH)

# Set default base image dynamically for each arch
ifeq ($(ARCH),amd64)
        BASEIMAGE?=redis
endif
ifeq ($(ARCH),arm)
        BASEIMAGE?=armhf/redis
endif
ifeq ($(ARCH),arm64)
        BASEIMAGE?=aarch64/redis
endif
ifeq ($(ARCH),ppc64le)
        BASEIMAGE?=ppc64le/redis
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

container: .container-$(ARCH)
.container-$(ARCH):
	cp ./* $(TEMP_DIR)
	# Set the base image
	cd $(TEMP_DIR) && sed -i.bak 's|BASEIMAGE|$(BASEIMAGE)|g' Dockerfile
	docker build --pull -t $(MULTI_ARCH_IMG):$(TAG) $(TEMP_DIR)

ifeq ($(ARCH), amd64)
	# This is for to maintain the backward compatibility
	docker tag $(MULTI_ARCH_IMG):$(TAG) $(IMAGE):$(TAG)
endif

push: .push-$(ARCH)
.push-$(ARCH): .container-$(ARCH)
	gcloud docker -- push $(MULTI_ARCH_IMG):$(TAG)

ifeq ($(ARCH),amd64)
	gcloud docker -- push $(IMAGE):$(TAG)
endif

clean:
	rm -rf $(BIN)
