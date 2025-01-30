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

.PHONY: all container clean orphan all-push push-manifest

REGISTRY ?= staging-k8s.gcr.io
IMAGE = $(REGISTRY)/pause

TAG ?= 3.10
REV = $(shell git describe --contains --always --match='v*')

# Architectures supported: amd64, arm, arm64, ppc64le and s390x
ARCH ?= amd64
# Operating systems supported: linux, windows
OS ?= linux
# OS Version for the Windows images: 1809, ltsc2022
OSVERSION ?= 1809 ltsc2022

# The output type could either be docker (local), or registry.
# If it is registry, it will also allow us to push the Windows images.
OUTPUT_TYPE ?= docker

ALL_OS = linux windows
ALL_ARCH.linux = amd64 arm arm64 ppc64le s390x
ALL_OS_ARCH.linux = $(foreach arch, ${ALL_ARCH.linux}, linux-$(arch))
ALL_ARCH.windows = amd64
ALL_OSVERSIONS.windows := 1809 ltsc2022
ALL_OS_ARCH.windows = $(foreach arch, $(ALL_ARCH.windows), $(foreach osversion, ${ALL_OSVERSIONS.windows}, windows-$(arch)-${osversion}))
ALL_OS_ARCH = $(foreach os, $(ALL_OS), ${ALL_OS_ARCH.${os}})

CFLAGS = -Os -Wall -Werror -static -DVERSION=v$(TAG)-$(REV)
KUBE_CROSS_IMAGE ?= registry.k8s.io/build-image/kube-cross
KUBE_CROSS_VERSION ?= $(shell cat ../build-image/cross/VERSION)

# NOTE(claudiub): The Windows pause image also requires the wincat binary we're compiling for the
# port-forwarding scenarios. If it's no longer necessary, it can be removed.
# For more information, see: https://github.com/kubernetes/kubernetes/pull/91452
BIN.linux = pause
BIN.windows = pause wincat
BIN := ${BIN.${OS}}
SRCS.linux = linux/pause.c
SRCS.windows = windows/pause.c
SRCS := ${SRCS.${OS}}

EXTENSION.linux =
EXTENSION.windows = .exe
EXTENSION := ${EXTENSION.${OS}}

# The manifest command is still experimental as of Docker 18.09.3
export DOCKER_CLI_EXPERIMENTAL=enabled

TRIPLE.windows-amd64 := x86_64-w64-mingw32
TRIPLE.linux-amd64 := x86_64-linux-gnu
TRIPLE.linux-arm := arm-linux-gnueabihf
TRIPLE.linux-arm64 := aarch64-linux-gnu
TRIPLE.linux-ppc64le := powerpc64le-linux-gnu
TRIPLE.linux-s390x := s390x-linux-gnu
TRIPLE := ${TRIPLE.${OS}-${ARCH}}
BASE.linux := scratch
# Source for windows pause image base is located at https://github.com/microsoft/windows-pause-image-base
BASE.windows := mcr.microsoft.com/oss/kubernetes/windows-pause-image-base:v0.2
BASE := ${BASE.${OS}}

# If you want to build AND push all containers, see the 'all-push' rule.
all: all-container-docker

# NOTE(claudiub): A non-default builder instance is needed in order to build Windows images.
all-push: all-container-registry push-manifest

push-manifest: SHELL:=/bin/bash
push-manifest:
	docker manifest create --amend $(IMAGE):$(TAG) $(shell echo $(ALL_OS_ARCH) | sed -e "s~[^ ]*~$(IMAGE):$(TAG)\-&~g")
	set -x; for arch in $(ALL_ARCH.linux); do docker manifest annotate --os linux --arch $${arch} ${IMAGE}:${TAG} ${IMAGE}:${TAG}-linux-$${arch}; done
	# For Windows images, we also need to include the "os.version" in the manifest list, so the Windows node can pull the proper image it needs.
	# we use awk to also trim the quotes around the OS version string.
	set -x; \
	# tagToKernelVersionMap maps the container images tags for different Windows Server releases (ex: ltsc2022 for Windows Server 2022)
	# to the kernel version for that OS release (ex: 20348 for Windows Server 2022). This is needed to fetch the servicing revision from the
	# pause base image manifest (which containers an entry for each Windows Server version) so we can add the approrite 'os.version'
	# field to the pause image manifest.
	declare -A tagToKernelVersionMap=( ['1809']='17763' ['ltsc2022']='20348' );\
	for arch in $(ALL_ARCH.windows);  do \
		for osversion in ${ALL_OSVERSIONS.windows}; do \
			full_version=`docker manifest inspect ${BASE.windows} | grep "10.0.$${tagToKernelVersionMap[$$osversion]}" | head -n 1 | awk -F\" '{print $$4}'` || true; \
			docker manifest annotate --os windows --arch $${arch} --os-version $${full_version} ${IMAGE}:${TAG} ${IMAGE}:${TAG}-windows-$${arch}-$${osversion}; \
		done; \
	done
	docker manifest push --purge ${IMAGE}:${TAG}

all-container-docker: $(addprefix sub-container-docker-,$(ALL_OS_ARCH.linux))
all-container-registry: $(addprefix sub-container-registry-,$(ALL_OS_ARCH))

# split words on hyphen, access by 1-index
word-hyphen = $(word $2,$(subst -, ,$1))
sub-container-%:
	$(MAKE) OUTPUT_TYPE=$(call word-hyphen,$*,1) OS=$(call word-hyphen,$*,2) ARCH=$(call word-hyphen,$*,3) OSVERSION=$(call word-hyphen,$*,4) container

build: $(foreach binary, ${BIN}, bin/${binary}-${OS}-${ARCH})

bin/${BIN.linux}-$(OS)-$(ARCH): $(SRCS)
	mkdir -p bin
	docker run --rm -u $$(id -u):$$(id -g) -v $$(pwd):/build \
		$(KUBE_CROSS_IMAGE):$(KUBE_CROSS_VERSION) \
		/bin/bash -c "\
			cd /build && \
			$(TRIPLE)-gcc $(CFLAGS) -o $@ $^ && \
			$(TRIPLE)-strip $(foreach binary, $@, ${binary}${EXTENSION})"

bin/wincat-windows-${ARCH}: windows/wincat/wincat.go
	mkdir -p bin
	docker run --rm -u $$(id -u):$$(id -g) -v $$(pwd):/build \
		--tmpfs /.cache \
		$(KUBE_CROSS_IMAGE):$(KUBE_CROSS_VERSION) \
		/bin/bash -c "\
		cd /build && \
		CGO_ENABLED=0 GOOS=windows GOARCH=${ARCH} go build -o $@ $^"

container: .container-${OS}-$(ARCH)
.container-linux-$(ARCH): bin/$(BIN)-$(OS)-$(ARCH)
	docker buildx build --provenance=false --sbom=false --pull --output=type=${OUTPUT_TYPE} --platform ${OS}/$(ARCH) \
		-t $(IMAGE):$(TAG)-${OS}-$(ARCH) --build-arg BASE=${BASE} --build-arg ARCH=$(ARCH) .
	touch $@

.container-windows-$(ARCH): $(foreach binary, ${BIN}, bin/${binary}-${OS}-${ARCH})
	docker buildx build --provenance=false --sbom=false --pull --output=type=${OUTPUT_TYPE} --platform ${OS}/$(ARCH) \
		-t $(IMAGE):$(TAG)-${OS}-$(ARCH)-${OSVERSION} --build-arg BASE=${BASE}-windows-${OSVERSION}-${ARCH} --build-arg ARCH=$(ARCH) -f Dockerfile_windows .
	touch $@

# Useful for testing, not automatically included in container image
orphan: bin/orphan-linux-$(ARCH)
bin/orphan-linux-$(ARCH): linux/orphan.c
	mkdir -p bin
	docker run --rm -u $$(id -u):$$(id -g) -v $$(pwd):/build \
		$(KUBE_CROSS_IMAGE):$(KUBE_CROSS_VERSION) \
		/bin/bash -c "\
			cd /build && \
			$(TRIPLE)-gcc $(CFLAGS) -o $@ $^ && \
			$(TRIPLE)-strip $@"

clean:
	rm -rf .*container-* .push-* bin/
