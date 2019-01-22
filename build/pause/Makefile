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

.PHONY: all push container clean orphan all-push push-manifest

include ../../hack/make-rules/Makefile.manifest

REGISTRY ?= staging-k8s.gcr.io
IMAGE = $(REGISTRY)/pause
IMAGE_WITH_ARCH = $(IMAGE)-$(ARCH)

TAG = 3.1
REV = $(shell git describe --contains --always --match='v*')

# Architectures supported: amd64, arm, arm64, ppc64le and s390x
ARCH ?= amd64

ALL_ARCH = amd64 arm arm64 ppc64le s390x

CFLAGS = -Os -Wall -Werror -static -DVERSION=v$(TAG)-$(REV)
KUBE_CROSS_IMAGE ?= k8s.gcr.io/kube-cross
KUBE_CROSS_VERSION ?= $(shell cat ../build-image/cross/VERSION)

BIN = pause
SRCS = pause.c

ifeq ($(ARCH),amd64)
	TRIPLE ?= x86_64-linux-gnu
endif

ifeq ($(ARCH),arm)
	TRIPLE ?= arm-linux-gnueabihf
endif

ifeq ($(ARCH),arm64)
	TRIPLE ?= aarch64-linux-gnu
endif

ifeq ($(ARCH),ppc64le)
	TRIPLE ?= powerpc64le-linux-gnu
endif

ifeq ($(ARCH),s390x)
	TRIPLE ?= s390x-linux-gnu
endif

# If you want to build AND push all containers, see the 'all-push' rule.
all: all-container

all-push: all-push-images push-manifest

push-manifest: manifest-tool
	manifest-tool push from-args --platforms $(call join_platforms,$(ALL_ARCH)) --template $(IMAGE)-ARCH:$(TAG) --target $(IMAGE):$(TAG)

sub-container-%:
	$(MAKE) ARCH=$* container

sub-push-%:
	$(MAKE) ARCH=$* push

all-container: $(addprefix sub-container-,$(ALL_ARCH))

all-push-images: $(addprefix sub-push-,$(ALL_ARCH))

build: bin/$(BIN)-$(ARCH)

bin/$(BIN)-$(ARCH): $(SRCS)
	mkdir -p bin
	docker run --rm -u $$(id -u):$$(id -g) -v $$(pwd):/build \
		$(KUBE_CROSS_IMAGE):$(KUBE_CROSS_VERSION) \
		/bin/bash -c "\
			cd /build && \
			$(TRIPLE)-gcc $(CFLAGS) -o $@ $^ && \
			$(TRIPLE)-strip $@"

container: .container-$(ARCH)
.container-$(ARCH): bin/$(BIN)-$(ARCH)
	docker build --pull -t $(IMAGE_WITH_ARCH):$(TAG) --build-arg ARCH=$(ARCH) .
	touch $@

push: .push-$(ARCH)
.push-$(ARCH): .container-$(ARCH)
	docker push $(IMAGE_WITH_ARCH):$(TAG)
	touch $@

# Useful for testing, not automatically included in container image
orphan: bin/orphan-$(ARCH)
bin/orphan-$(ARCH): orphan.c
	mkdir -p bin
	docker run -u $$(id -u):$$(id -g) -v $$(pwd):/build \
		$(KUBE_CROSS_IMAGE):$(KUBE_CROSS_VERSION) \
		/bin/bash -c "\
			cd /build && \
			$(TRIPLE)-gcc $(CFLAGS) -o $@ $^ && \
			$(TRIPLE)-strip $@"

clean:
	rm -rf .container-* .push-* bin/
