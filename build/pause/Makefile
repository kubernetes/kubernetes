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
IMAGE_WITH_OS_ARCH = $(IMAGE)-$(OS)-$(ARCH)

TAG = 3.4
REV = $(shell git describe --contains --always --match='v*')

# Architectures supported: amd64, arm, arm64, ppc64le and s390x
ARCH ?= amd64
# Operating systems supported: linux, windows
OS ?= linux
# OS Version for the Windows images: 1809, 1903, 1909 2004
OSVERSION ?= 1809 1903 1909 2004

# The output type could either be docker (local), or registry.
# If it is registry, it will also allow us to push the Windows images.
OUTPUT_TYPE ?= docker

ALL_OS = linux windows
ALL_ARCH.linux = amd64 arm arm64 ppc64le s390x
ALL_OS_ARCH.linux = $(foreach arch, ${ALL_ARCH.linux}, linux-$(arch))
ALL_ARCH.windows = amd64
ALL_OSVERSIONS.windows := 1809 1903 1909 2004
ALL_OS_ARCH.windows = $(foreach arch, $(ALL_ARCH.windows), $(foreach osversion, ${ALL_OSVERSIONS.windows}, windows-$(arch)-${osversion}))
ALL_OS_ARCH = $(foreach os, $(ALL_OS), ${ALL_OS_ARCH.${os}})

CFLAGS = -Os -Wall -Werror -static -DVERSION=v$(TAG)-$(REV)
KUBE_CROSS_IMAGE.linux ?= k8s.gcr.io/build-image/kube-cross
KUBE_CROSS_VERSION.linux ?= $(shell cat ../build-image/cross/VERSION)
KUBE_CROSS_IMAGE.windows ?= dockcross/windows-static-x64
KUBE_CROSS_VERSION.windows ?= latest
KUBE_CROSS_IMAGE := ${KUBE_CROSS_IMAGE.${OS}}
KUBE_CROSS_VERSION := ${KUBE_CROSS_VERSION.${OS}}

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

TRIPLE.windows-amd64 := x86_64-w64-mingw32.static
TRIPLE.linux-amd64 := x86_64-linux-gnu
TRIPLE.linux-arm := arm-linux-gnueabihf
TRIPLE.linux-arm64 := aarch64-linux-gnu
TRIPLE.linux-ppc64le := powerpc64le-linux-gnu
TRIPLE.linux-s390x := s390x-linux-gnu
TRIPLE := ${TRIPLE.${OS}-${ARCH}}
BASE.linux := scratch
BASE.windows := mcr.microsoft.com/windows/nanoserver
BASE := ${BASE.${OS}}

# If you want to build AND push all containers, see the 'all-push' rule.
all: all-container-docker

# NOTE(claudiub): A non-default builder instance is needed in order to build Windows images.
all-push: all-container-registry push-manifest

push-manifest:
	docker manifest create --amend $(IMAGE):$(TAG) $(shell echo $(ALL_OS_ARCH) | sed -e "s~[^ ]*~$(IMAGE)\-&:$(TAG)~g")
	set -x; for arch in $(ALL_ARCH.linux); do docker manifest annotate --os linux --arch $${arch} ${IMAGE}:${TAG} ${IMAGE}-linux-$${arch}:${TAG}; done
	# For Windows images, we also need to include the "os.version" in the manifest list, so the Windows node can pull the proper image it needs.
	# At the moment, docker manifest annotate doesn't allow us to set the os.version, so we'll have to it ourselves. The manifest list can be found locally as JSONs.
	# See: https://github.com/moby/moby/issues/41417
	# If the ${REGISTRY} is on dockerhub, the prefix "docker.io/" is optional. However, we need the full
	# registry name for setting the os.version for Windows images below.
	# If the registry name does not contain any slashes, we prepend "docker.io/"
	# TODO(claudiub): Clean this up once the above issue has been fixed.
	set -x; \
	registry_prefix=$(shell (echo ${REGISTRY} | grep -Eq ".*\/.*") && echo "" || echo "docker.io/"); \
	manifest_image_folder=`echo "$${registry_prefix}${IMAGE}" | sed "s|/|_|g" | sed "s/:/-/"`; \
	for arch in $(ALL_ARCH.windows);  do \
		for osversion in ${ALL_OSVERSIONS.windows}; do \
			docker manifest annotate --os windows --arch $${arch} ${IMAGE}:${TAG} ${IMAGE}-windows-$${arch}-$${osversion}:${TAG}; \
			BASEIMAGE=${BASE.windows}:$${osversion}; \
			full_version=`docker manifest inspect ${BASE.windows}:$${osversion} | grep "os.version" | head -n 1 | awk '{print $$2}'` || true; \
			sed -i -r "s/(\"os\"\:\"windows\")/\0,\"os.version\":$${full_version}/" "${HOME}/.docker/manifests/$${manifest_image_folder}-${TAG}/$${manifest_image_folder}-windows-$${arch}-$${osversion}-${TAG}"; \
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
	CGO_ENABLED=0 GOOS=windows GOARCH=${ARCH} go build -o $@ $^

container: .container-${OS}-$(ARCH)
.container-linux-$(ARCH): bin/$(BIN)-$(OS)-$(ARCH)
	docker buildx build --pull --output=type=${OUTPUT_TYPE} --platform ${OS}/$(ARCH) \
		-t $(IMAGE_WITH_OS_ARCH):$(TAG) --build-arg BASE=${BASE} --build-arg ARCH=$(ARCH) .
	touch $@

.container-windows-$(ARCH): $(foreach binary, ${BIN}, bin/${binary}-${OS}-${ARCH})
	docker buildx build --pull --output=type=${OUTPUT_TYPE} --platform ${OS}/$(ARCH) \
		-t $(IMAGE_WITH_OS_ARCH)-${OSVERSION}:$(TAG) --build-arg BASE=${BASE}:${OSVERSION} --build-arg ARCH=$(ARCH) -f Dockerfile_windows .
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

clean:
	rm -rf .*-container-* .push-* bin/
