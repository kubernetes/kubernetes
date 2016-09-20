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

# Build the flannel image
#
# Usage:
# 	[TAG=?] [REGISTRY=gcr.io/google_containers] [ARCH=amd64] make build

TAG?=0.6.2-kube
ARCH?=amd64
REGISTRY?=gcr.io/google_containers
TEMP_DIR:=$(shell mktemp -d)
BASEIMAGE?=gcr.io/google_containers/debian-iptables-${ARCH}:v2
ARCHES = amd64 arm arm64 ppc64le
FLANNEL_COMMIT=44bac89c8065fa2961b35ca094573f8db92456f5

all: build-all
build-all: $(addprefix build-,$(ARCHES))
push-all: $(addprefix push-,$(ARCHES))

$(TEMP_DIR)/flannel:
	git clone https://github.com/coreos/flannel $(TEMP_DIR)/flannel
	cd $(TEMP_DIR)/flannel && git checkout $(FLANNEL_COMMIT)

build-%: $(TEMP_DIR)/flannel

	# Build the image with flannel's official scripts
	cd $(TEMP_DIR)/flannel && ARCH=$* TAG=$(TAG) make dist/flanneld-$(TAG)-$*.docker

	# And retag the image to gcr.io for pushing
	docker tag quay.io/coreos/flannel:$(TAG)-$* $(REGISTRY)/flannel-$*:$(TAG)

push-%: build-$*
	gcloud docker push $(REGISTRY)/flannel-$(ARCH):$(TAG)
