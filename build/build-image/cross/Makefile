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

include ../../../hack/make-rules/Makefile.manifest

REGISTRY ?= staging-k8s.gcr.io
IMAGE = $(REGISTRY)/kube-cross
IMAGE_WITH_ARCH = $(IMAGE)-$(ARCH)

# This option is for running docker manifest command
export DOCKER_CLI_EXPERIMENTAL := enabled

TAG=$(shell cat VERSION)

# Architectures supported: amd64, arm64, ppc64le
ARCH ?= amd64

ALL_ARCH = amd64 arm64 ppc64le

TEMP_DIR:=$(shell mktemp -d)
QEMUVERSION=v2.9.1

ifeq ($(ARCH),arm64)
	QEMUARCH=aarch64
endif
ifeq ($(ARCH),ppc64le)
	QEMUARCH=ppc64le
endif

SUDO=$(if $(filter 0,$(shell id -u)),,sudo)

# If you want to build AND push all containers, see the 'all-push' rule.
all: all-container

all-push: all-push-images push-manifest

push-manifest:
	docker manifest create --amend $(IMAGE):$(TAG) $(shell echo $(ALL_ARCH) | sed -e "s~[^ ]*~$(IMAGE)\-&:$(TAG)~g")
	@for arch in $(ALL_ARCH); do docker manifest annotate --arch $${arch} ${IMAGE}:${TAG} ${IMAGE}-$${arch}:${TAG}; done
	docker manifest push --purge ${IMAGE}:${TAG}

sub-container-%:
	$(MAKE) ARCH=$* container

sub-push-%:
	$(MAKE) ARCH=$* push

all-container: $(addprefix sub-container-,$(ALL_ARCH))

all-push-images: $(addprefix sub-push-,$(ALL_ARCH))

container: .container-$(ARCH)
.container-$(ARCH):
	cp ./* $(TEMP_DIR)
	cat Dockerfile \
		| sed "s|QEMUARCH|$(QEMUARCH)|g" \
		> $(TEMP_DIR)/Dockerfile

ifeq ($(ARCH),amd64)
	# When building "normally" for amd64, remove the whole line, it has no part in the amd64 image
	sed "/CROSS_BUILD_/d" $(TEMP_DIR)/Dockerfile > $(TEMP_DIR)/Dockerfile.tmp
else
	# When cross-building, only the placeholder "CROSS_BUILD_" should be removed
	# Register /usr/bin/qemu-ARCH-static as the handler for non-x86 binaries in the kernel
	$(SUDO) ../../../third_party/multiarch/qemu-user-static/register/register.sh --reset
	curl -sSL https://github.com/multiarch/qemu-user-static/releases/download/$(QEMUVERSION)/x86_64_qemu-$(QEMUARCH)-static.tar.gz | tar -xz -C $(TEMP_DIR)
	# Ensure we don't get surprised by umask settings
	chmod 0755 $(TEMP_DIR)/qemu-$(QEMUARCH)-static
	sed "s/CROSS_BUILD_//g" $(TEMP_DIR)/Dockerfile > $(TEMP_DIR)/Dockerfile.tmp
endif
	mv $(TEMP_DIR)/Dockerfile.tmp $(TEMP_DIR)/Dockerfile

	docker build --pull -t $(IMAGE_WITH_ARCH):$(TAG) --build-arg ARCH=$(ARCH) -f $(TEMP_DIR)/Dockerfile $(TEMP_DIR)
	rm -rf $(TEMP_DIR)
	touch $@

push: .push-$(ARCH)
.push-$(ARCH): .container-$(ARCH)
	docker push $(IMAGE_WITH_ARCH):$(TAG)
	touch $@

clean:
	rm -rf .container-* .push-* bin/
