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

REGISTRY?=staging-k8s.gcr.io

IMAGE=$(REGISTRY)/kube-addon-manager
ARCH?=amd64
ALL_ARCH = amd64 arm arm64 ppc64le

export DOCKER_CLI_EXPERIMENTAL := enabled

TEMP_DIR:=$(shell mktemp -d)
VERSION=v8.9
KUBECTL_VERSION?=v1.11.3

BASEIMAGE=k8s.gcr.io/debian-base-$(ARCH):0.4.0

.PHONY: build push push-manifest all-push all-push-images all-build all all-push-images all-push push-manifest

all: all-build

build:
	cp ./* $(TEMP_DIR)
	curl -sSL --retry 5 https://dl.k8s.io/release/$(KUBECTL_VERSION)/bin/linux/$(ARCH)/kubectl > $(TEMP_DIR)/kubectl
	chmod +x $(TEMP_DIR)/kubectl
	cd $(TEMP_DIR) && sed -i.back "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile
	docker build --pull -t $(IMAGE)-$(ARCH):$(VERSION) $(TEMP_DIR)

push: build
	docker push $(IMAGE)-$(ARCH):$(VERSION)

sub-build-%:
	$(MAKE) ARCH=$* build

all-build: $(addprefix sub-build-,$(ALL_ARCH))

sub-push-image-%:
	$(MAKE) ARCH=$* push


all-push-images: $(addprefix sub-push-image-,$(ALL_ARCH))


all-push: all-push-images push-manifest


push-manifest:
	docker manifest create --amend $(IMAGE):$(VERSION) $(shell echo $(ALL_ARCH) | sed -e "s~[^ ]*~$(IMAGE)\-&:$(VERSION)~g")
	for arch in $(ALL_ARCH); do docker manifest annotate --arch $${arch} ${IMAGE}:${VERSION} ${IMAGE}-$${arch}:${VERSION}; done
	docker manifest push --purge ${IMAGE}:${VERSION}


clean:
	docker rmi -f $(IMAGE)-$(ARCH):$(VERSION)
