# Copyright 2017 The Kubernetes Authors.
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

all: build

REGISTRY ?= gcr.io/k8s-staging-build-image
IMAGE ?= $(REGISTRY)/debian-base

TAG ?= v2.2.0

BASEIMAGE ?= debian:buster-slim

# required to enable buildx
export DOCKER_CLI_EXPERIMENTAL := enabled

# build with buildx
PLATFORMS?=linux/amd64,linux/arm,linux/arm64,linux/ppc64le,linux/s390x
OUTPUT=
PROGRESS=plain

# build multi-arch image
build: ensure-buildx
	docker buildx build \
		--platform=${PLATFORMS} $(OUTPUT) \
		--progress=$(PROGRESS) \
		--build-arg BASEIMAGE="$(BASEIMAGE)" \
		-t $(IMAGE):$(TAG) \
		-f Dockerfile.build \
		--pull .

# push the cross built image
push: OUTPUT=--push
push: build

# standard cleanup target
clean:
	docker rmi -f $(IMAGE):$(TAG) || true
	docker rm  -f $(IMAGE):$(TAG) || true

# enable buildx
ensure-buildx:
	./../../hack/init-buildx.sh

.PHONY: build push clean ensure-buildx
