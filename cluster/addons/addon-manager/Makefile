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

IMAGE=gcr.io/google-containers/kube-addon-manager
ARCH?=amd64
TEMP_DIR:=$(shell mktemp -d)
VERSION=v5.1

# amd64 and arm has "stable" binaries pushed for v1.2, arm64 and ppc64le hasn't so they have to fetch the latest alpha
# however, arm64 and ppc64le are very experimental right now, so it's okay
ifeq ($(ARCH),amd64)
	KUBECTL_VERSION?=v1.3.0-beta.2
	BASEIMAGE?=python:2.7-slim
endif
ifeq ($(ARCH),arm)
	KUBECTL_VERSION?=v1.3.0-beta.2
	BASEIMAGE?=hypriot/rpi-python:2.7
	QEMUARCH=arm
endif
ifeq ($(ARCH),arm64)
	KUBECTL_VERSION?=v1.3.0-beta.2
	BASEIMAGE?=aarch64/python:2.7-slim
	QEMUARCH=aarch64
endif
ifeq ($(ARCH),ppc64le)
	KUBECTL_VERSION?=v1.3.0-beta.2
	BASEIMAGE?=ppc64le/python:2.7-slim
	QEMUARCH=ppc64le
endif

.PHONY: build push

all: build
build:
	cp ./* $(TEMP_DIR)
	curl -sSL --retry 5 https://storage.googleapis.com/kubernetes-release/release/$(KUBECTL_VERSION)/bin/linux/$(ARCH)/kubectl > $(TEMP_DIR)/kubectl
	chmod +x $(TEMP_DIR)/kubectl
	cd ${TEMP_DIR} && sed -i.back "s|ARCH|$(QEMUARCH)|g" Dockerfile
	cd $(TEMP_DIR) && sed -i.back "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

ifeq ($(ARCH),amd64)
	# When building "normally" for amd64, remove the whole line, it has no part in the amd64 image
	cd $(TEMP_DIR) && sed -i "/CROSS_BUILD_/d" Dockerfile
else
	# When cross-building, only the placeholder "CROSS_BUILD_" should be removed
	# Register /usr/bin/qemu-ARCH-static as the handler for other-arch binaries in the kernel
	docker run --rm --privileged multiarch/qemu-user-static:register --reset
	curl -sSL --retry 5 https://github.com/multiarch/qemu-user-static/releases/download/v2.5.0/x86_64_qemu-$(QEMUARCH)-static.tar.xz | tar -xJ -C $(TEMP_DIR)
	cd $(TEMP_DIR) && sed -i "s/CROSS_BUILD_//g" Dockerfile
endif

	docker build -t $(IMAGE)-$(ARCH):$(VERSION) $(TEMP_DIR)

push: build
	gcloud docker push $(IMAGE)-$(ARCH):$(VERSION)
ifeq ($(ARCH),amd64)
	# Backward compatibility. TODO: deprecate this image tag
	docker tag -f $(IMAGE)-$(ARCH):$(VERSION) $(IMAGE):$(VERSION)
	gcloud docker push $(IMAGE):$(VERSION)
endif

clean:
	docker rmi -f $(IMAGE)-$(ARCH):$(VERSION)
