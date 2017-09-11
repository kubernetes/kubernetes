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
VERSION=v6.4-beta.2
KUBECTL_VERSION?=v1.6.4

ifeq ($(ARCH),amd64)
	BASEIMAGE?=bashell/alpine-bash
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=arm32v7/debian
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=arm64v8/debian
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/debian
endif
ifeq ($(ARCH),s390x)
	BASEIMAGE?=s390x/debian
endif

.PHONY: build push

all: build

build:
	cp ./* $(TEMP_DIR)
	curl -sSL --retry 5 https://storage.googleapis.com/kubernetes-release/release/$(KUBECTL_VERSION)/bin/linux/$(ARCH)/kubectl > $(TEMP_DIR)/kubectl
	chmod +x $(TEMP_DIR)/kubectl
	cd $(TEMP_DIR) && sed -i.back "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile
	docker build --pull -t $(IMAGE)-$(ARCH):$(VERSION) $(TEMP_DIR)

push: build
	gcloud docker -- push $(IMAGE)-$(ARCH):$(VERSION)
ifeq ($(ARCH),amd64)
	# Backward compatibility. TODO: deprecate this image tag
	docker rmi $(IMAGE):$(VERSION) 2>/dev/null || true
	docker tag $(IMAGE)-$(ARCH):$(VERSION) $(IMAGE):$(VERSION)
	gcloud docker -- push $(IMAGE):$(VERSION)
endif

clean:
	docker rmi -f $(IMAGE)-$(ARCH):$(VERSION)
