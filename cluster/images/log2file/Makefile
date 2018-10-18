# Copyright 2018 The Kubernetes Authors.
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

# build Kubemark image from currently built binaries containing both 'real' master and Hollow Node.
# This makefile assumes that the kubemark binary is present in this directory.

# Allow the caller to override this.  Beware make's precedence. This:
#   REGISTRY=$VAR make
# .. is not the same as:
#   make REGISTRY=$VAR
REGISTRY ?= staging-k8s.gcr.io
IMAGE = $(REGISTRY)/log2file
TAG ?= latest
ARCH ?= amd64
ALL_ARCH = amd64 arm arm64 ppc64le s390x

# This option is for running docker manifest command
export DOCKER_CLI_EXPERIMENTAL := enabled

build:
	docker build --build-arg ARCH=$(ARCH) -t $(IMAGE)-$(ARCH):$(TAG) .

push:
	docker push $(IMAGE)-$(ARCH):$(TAG)

sub-build-%:
	$(MAKE) ARCH=$* build

all-build: $(addprefix sub-build-,$(ALL_ARCH))

sub-push-image-%:
	$(MAKE) ARCH=$* push

push-manifest: $(addprefix sub-push-image-,$(ALL_ARCH))
	docker manifest create --amend $(IMAGE):$(TAG) $(shell echo $(ALL_ARCH) | sed -e "s~[^ ]*~$(IMAGE)\-&:$(TAG)~g")
	@for arch in $(ALL_ARCH); do docker manifest annotate --arch $${arch} ${IMAGE}:${TAG} ${IMAGE}-$${arch}:${TAG}; done
	docker manifest push ${IMAGE}:${TAG}
