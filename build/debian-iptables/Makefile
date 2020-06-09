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

.PHONY:	build push all all-build all-push-images all-push push-manifest

REGISTRY?="gcr.io/k8s-staging-build-image"
IMAGE=$(REGISTRY)/debian-iptables
TAG?=v12.1.0
ARCH?=amd64
ALL_ARCH = amd64 arm arm64 ppc64le s390x
TEMP_DIR:=$(shell mktemp -d)

BASE_REGISTRY?=us.gcr.io/k8s-artifacts-prod/build-image
BASEIMAGE?=$(BASE_REGISTRY)/debian-base-$(ARCH):v2.1.0

# This option is for running docker manifest command
export DOCKER_CLI_EXPERIMENTAL := enabled

SUDO=$(if $(filter 0,$(shell id -u)),,sudo)

build:
	cp ./* $(TEMP_DIR)
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

ifneq ($(ARCH),amd64)
	# Register /usr/bin/qemu-ARCH-static as the handler for non-x86 binaries in the kernel
	$(SUDO) ../../third_party/multiarch/qemu-user-static/register/register.sh --reset
endif

	docker build --pull -t $(IMAGE)-$(ARCH):$(TAG) $(TEMP_DIR)

push: build
	docker push $(IMAGE)-$(ARCH):$(TAG)

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

all: all-push
