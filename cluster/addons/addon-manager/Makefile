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

IMAGE=gcr.io/k8s-staging-addon-manager/kube-addon-manager
ARCH?=amd64
TEMP_DIR:=$(shell mktemp -d)
VERSION=v9.1.6
KUBECTL_VERSION?=v1.20.2

BASEIMAGE=registry.k8s.io/debian-base-$(ARCH):v1.0.1

SUDO=$(if $(filter 0,$(shell id -u)),,sudo)

.PHONY: build push

all: build

build:
	cp ./* $(TEMP_DIR)
	curl -sSL --retry 5 https://dl.k8s.io/release/$(KUBECTL_VERSION)/bin/linux/$(ARCH)/kubectl > $(TEMP_DIR)/kubectl
	chmod a+rx $(TEMP_DIR)/kube-addons.sh $(TEMP_DIR)/kube-addons-main.sh $(TEMP_DIR)/kubectl

ifneq ($(ARCH),amd64)
	# Register /usr/bin/qemu-ARCH-static as the handler for non-x86 binaries in the kernel
	$(SUDO) ../../../third_party/multiarch/qemu-user-static/register/register.sh --reset
endif

	docker build --pull -t $(IMAGE)-$(ARCH):$(VERSION) $(TEMP_DIR) --build-arg BASEIMAGE=$(BASEIMAGE)

push: build
	docker push $(IMAGE)-$(ARCH):$(VERSION)
ifeq ($(ARCH),amd64)
	# Backward compatibility. TODO: deprecate this image tag
	docker rmi $(IMAGE):$(VERSION) 2>/dev/null || true
	docker tag $(IMAGE)-$(ARCH):$(VERSION) $(IMAGE):$(VERSION)
	docker push $(IMAGE):$(VERSION)
endif

test:
	cp ./* $(TEMP_DIR)
	curl -sSL --retry 5 https://dl.k8s.io/release/$(KUBECTL_VERSION)/bin/linux/$(ARCH)/kubectl > $(TEMP_DIR)/kubectl
	chmod a+rx $(TEMP_DIR)/kube-addons.sh $(TEMP_DIR)/kube-addons-test.sh $(TEMP_DIR)/kubectl
	cd $(TEMP_DIR) && KUBECTL_BIN=$(TEMP_DIR)/kubectl ./kube-addons-test.sh

clean:
	docker rmi -f $(IMAGE)-$(ARCH):$(VERSION)
