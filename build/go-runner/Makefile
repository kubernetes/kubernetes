# Copyright 2020 The Kubernetes Authors.
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

# set default shell
SHELL=/bin/bash -o pipefail

TAG ?= v0.1.1
REGISTRY ?= k8s.gcr.io

IMGNAME = go-runner
IMAGE = $(REGISTRY)/$(IMGNAME)

PLATFORMS = linux/amd64 linux/arm64 linux/arm linux/ppc64le linux/s390x

HOST_GOOS ?= $(shell go env GOOS)
HOST_GOARCH ?= $(shell go env GOARCH)
GO_BUILD ?= go build

.PHONY: all build clean

.PHONY: all
all: build

.PHONY: build
build:
	$(GO_BUILD)

.PHONY: clean
clean:
	rm go-runner

.PHONY: container
container: init-docker-buildx
	# https://github.com/docker/buildx/issues/59
	$(foreach PLATFORM,$(PLATFORMS), \
		DOCKER_CLI_EXPERIMENTAL=enabled docker buildx build \
		--load \
		--progress plain \
		--platform $(PLATFORM) \
		--tag $(IMAGE)-$(PLATFORM):$(TAG) .;)

.PHONY: push
push: container
	$(foreach PLATFORM,$(PLATFORMS), \
		docker push $(IMAGE)-$(PLATFORM):$(TAG);)

.PHONY: manifest
manifest: push
	docker manifest create --amend $(IMAGE):$(TAG) $(shell echo $(PLATFORMS) | sed -e "s~[^ ]*~$(IMAGE)\-&:$(TAG)~g")
	@for arch in $(PLATFORMS); do docker manifest annotate --arch "$${arch##*/}" ${IMAGE}:${TAG} ${IMAGE}-$${arch}:${TAG}; done
	docker manifest push --purge $(IMAGE):$(TAG)

.PHONY: init-docker-buildx
init-docker-buildx:
ifneq ($(shell docker buildx 2>&1 >/dev/null; echo $?),)
	$(error "buildx not vailable. Docker 19.03 or higher is required")
endif
	docker run --rm --privileged linuxkit/binfmt:4ea3b9b0938cbd19834c096aa31ff475cc75d281
	docker buildx create --name multiarch-go-runner --use || true
	docker buildx inspect --bootstrap
