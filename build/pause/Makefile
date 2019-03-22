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

.PHONY: all push container clean orphan all-push push-manifest annotate manifest

REGISTRY ?= staging-k8s.gcr.io
IMAGE = $(REGISTRY)/pause
IMAGE_WITH_OS_ARCH = $(IMAGE)-$(OS)-$(ARCH)

TAG = 3.1
REV = $(shell git describe --contains --always --match='v*')

# Architectures supported: amd64, arm, arm64, ppc64le and s390x
ARCH ?= amd64
# Operating systems supported: linux, windows
OS ?= linux

ALL_OS = linux windows
ALL_ARCH.linux = amd64 arm arm64 ppc64le s390x
ALL_ARCH.windows = amd64
ALL_OS_ARCH = $(foreach os, $(ALL_OS), $(foreach arch, ${ALL_ARCH.${os}}, $(os)-$(arch)))

CFLAGS = -Os -Wall -Werror -static -DVERSION=v$(TAG)-$(REV)
KUBE_CROSS_IMAGE.linux ?= k8s.gcr.io/kube-cross
KUBE_CROSS_VERSION.linux ?= $(shell cat ../build-image/cross/VERSION)
KUBE_CROSS_IMAGE.windows ?= dockcross/windows-static-x64
KUBE_CROSS_VERSION.windows ?= latest
KUBE_CROSS_IMAGE := ${KUBE_CROSS_IMAGE.${OS}}
KUBE_CROSS_VERSION := ${KUBE_CROSS_VERSION.${OS}}

BIN = pause
SRCS.linux = linux/pause.c
SRCS.windows = windows/pause.c
SRCS := ${SRCS.${OS}}

# The manifest command is still experimental as of Docker 18.09.3
export DOCKER_CLI_EXPERIMENTAL=enabled

TRIPLE.linux-amd64 := x86_64-linux-gnu
TRIPLE.windows-amd64 := x86_64-w64-mingw32.static
TRIPLE.linux-arm := arm-linux-gnueabihf
TRIPLE.linux-arm64 := aarch64-linux-gnu
TRIPLE.linux-ppc64le := powerpc64le-linux-gnu
TRIPLE.linux-s390x := s390x-linux-gnu
TRIPLE := ${TRIPLE.${OS}-${ARCH}}

BASE.linux := scratch
BASE.windows := mcr.microsoft.com/windows/nanoserver:1809
BASE := ${BASE.${OS}}

# If you want to build AND push all containers, see the 'all-push' rule.
all: all-container

all-push: push-manifest

push-manifest: all-push-images all-annotate
	docker manifest push $(IMAGE):$(TAG)

# split words on hyphen, access by 1-index
word-hyphen = $(word $2,$(subst -, ,$1))
sub-%:
	$(MAKE) OS=$(call word-hyphen,$*,2) ARCH=$(call word-hyphen,$*,3) $(call word-hyphen,$*,1)

# targets prefixed with .windows will use the remote docker host
.windows-%: DOCKER_HOST=${WINDOWS_DOCKER_HOST}
.windows-%: DOCKER_CERT_PATH=${WINDOWS_DOCKER_CERT_PATH}
.windows-%: DOCKER_TLS_VERIFY=${WINDOWS_DOCKER_TLS_VERIFY}

all-annotate: $(addprefix sub-annotate-,$(ALL_OS_ARCH))
all-container: $(addprefix sub-container-,$(ALL_OS_ARCH))
all-push-images: $(addprefix sub-push-,$(ALL_OS_ARCH))

build: bin/$(BIN)-$(OS)-$(ARCH)

bin/$(BIN)-$(OS)-$(ARCH): $(SRCS)
	mkdir -p bin
	docker run --rm -u $$(id -u):$$(id -g) -v $$(pwd):/build \
		$(KUBE_CROSS_IMAGE):$(KUBE_CROSS_VERSION) \
		/bin/bash -c "\
			cd /build && \
			$(TRIPLE)-gcc $(CFLAGS) -o $@ $^ && \
			$(TRIPLE)-strip $@"

container: .$(OS)-container-$(ARCH)
.$(OS)-container-$(ARCH): bin/$(BIN)-$(OS)-$(ARCH)
	DOCKER_HOST=$(DOCKER_HOST) \
		DOCKER_CERT_PATH=$(DOCKER_CERT_PATH) \
		DOCKER_TLS_VERIFY=$(DOCKER_TLS_VERIFY) \
		docker build --pull -t $(IMAGE_WITH_OS_ARCH):$(TAG) \
		--build-arg BASE=$(BASE) --build-arg OS=$(OS) --build-arg ARCH=$(ARCH) .
	touch $@

push: .$(OS)-push-$(ARCH)
.$(OS)-push-$(ARCH): .$(OS)-container-$(ARCH)
	DOCKER_HOST=$(DOCKER_HOST) \
		DOCKER_CERT_PATH=$(DOCKER_CERT_PATH) \
		DOCKER_TLS_VERIFY=$(DOCKER_TLS_VERIFY) \
		docker push $(IMAGE_WITH_OS_ARCH):$(TAG)
	touch $@

# Useful for testing, not automatically included in container image
orphan: bin/orphan-linux-$(ARCH)
bin/orphan-linux-$(ARCH): linux/orphan.c
	mkdir -p bin
	docker run -u $$(id -u):$$(id -g) -v $$(pwd):/build \
		$(KUBE_CROSS_IMAGE):$(KUBE_CROSS_VERSION) \
		/bin/bash -c "\
			cd /build && \
			$(TRIPLE)-gcc $(CFLAGS) -o $@ $^ && \
			$(TRIPLE)-strip $@"

annotate: .$(OS)-annotate-$(ARCH)
.$(OS)-annotate-$(ARCH): .$(OS)-container-$(ARCH) .$(OS)-push-$(ARCH) .manifest
	docker manifest create --amend $(IMAGE):$(TAG) $(IMAGE_WITH_OS_ARCH):$(TAG)
	docker manifest annotate $(IMAGE):$(TAG) $(IMAGE_WITH_OS_ARCH):$(TAG) --os $(OS) --arch $(ARCH)
	touch $@

manifest: .manifest
.manifest:
	docker manifest create $(IMAGE):$(TAG) $(IMAGE_WITH_OS_ARCH):$(TAG)
	touch $@

clean:
	rm -rf .container-* .push-* .annotate-* .manifest bin/
