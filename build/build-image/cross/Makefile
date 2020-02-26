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

.PHONY:	build push

STAGING_REGISTRY?=gcr.io/k8s-staging-build-image
PROD_REGISTRY?=us.gcr.io/k8s-artifacts-prod/build-image
IMAGE=kube-cross

TAG?=kubernetes-$(shell git describe --tags --match='v*' --abbrev=14)
KUBE_CROSS_VERSION=$(shell cat VERSION)

all: build push

build:
	docker build \
		-t $(STAGING_REGISTRY)/$(IMAGE):$(TAG) \
		-t $(STAGING_REGISTRY)/$(IMAGE):$(KUBE_CROSS_VERSION) \
		-t $(PROD_REGISTRY)/$(IMAGE):$(KUBE_CROSS_VERSION) \
		.

push:
	docker push $(STAGING_REGISTRY)/$(IMAGE):$(TAG)
	docker push $(STAGING_REGISTRY)/$(IMAGE):$(KUBE_CROSS_VERSION)
