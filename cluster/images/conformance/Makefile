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

# Build the conformance image.
#
# Usage:
#   [ARCH=amd64] [REGISTRY="k8s.gcr.io"] make (build|push) VERSION={some_released_version_of_kubernetes}

REGISTRY?=k8s.gcr.io
OS?=linux
OS_VERSION?=1809
ARCH?=amd64
OUT_DIR?=_output

# The output type could either be docker (local), or registry.
# If it is registry, it will also allow us to push the Windows images.
OUTPUT_TYPE = docker
ifeq ($(MAKECMDGOALS),push)
  OUTPUT_TYPE = registry
else ifeq ($(OS),windows)
    $(error Windows images can only be created and pushed directly. Use make push for Windows images)
endif

IMAGE_NAME_EXTENSION.linux = $(OS)-$(ARCH)
IMAGE_NAME_EXTENSION.windows = $(OS)-$(ARCH)-$(OS_VERSION)
IMAGE_NAME_EXTENSION := ${IMAGE_NAME_EXTENSION.${OS}}

EXTENSION.linux =
EXTENSION.windows = .exe
EXTENSION := ${EXTENSION.${OS}}

GINKGO := ginkgo$(EXTENSION)
KUBECTL := kubectl$(EXTENSION)
E2E_TEST := e2e.test$(EXTENSION)
GO_RUNNER := go-runner$(EXTENSION)
GORUNNER := gorunner$(EXTENSION)

LOCAL_OUTPUT_PATH=$(shell pwd)/../../../$(OUT_DIR)/local/bin/$(OS)/$(ARCH)
DOCKERIZED_OUTPUT_PATH=$(shell pwd)/../../../$(OUT_DIR)/dockerized/bin/$(OS)/$(ARCH)

GINKGO_BIN?=$(shell test -f $(LOCAL_OUTPUT_PATH)/$(GINKGO) && echo $(LOCAL_OUTPUT_PATH)/$(GINKGO) || echo $(DOCKERIZED_OUTPUT_PATH)/$(GINKGO))
KUBECTL_BIN?=$(shell test -f $(LOCAL_OUTPUT_PATH)/$(KUBECTL) && echo $(LOCAL_OUTPUT_PATH)/$(KUBECTL) || echo $(DOCKERIZED_OUTPUT_PATH)/$(KUBECTL))
E2E_TEST_BIN?=$(shell test -f $(LOCAL_OUTPUT_PATH)/$(E2E_TEST) && echo $(LOCAL_OUTPUT_PATH)/$(E2E_TEST) || echo $(DOCKERIZED_OUTPUT_PATH)/$(E2E_TEST))
E2E_GO_RUNNER_BIN?=$(shell test -f $(LOCAL_OUTPUT_PATH)/$(GO_RUNNER) && echo $(LOCAL_OUTPUT_PATH)/$(GO_RUNNER) || echo $(DOCKERIZED_OUTPUT_PATH)/$(GO_RUNNER))

CLUSTER_DIR?=$(shell pwd)/../../../cluster/

BASEIMAGE.linux=debian:stable-slim
BASEIMAGE.windows=mcr.microsoft.com/windows/nanoserver:$(OS_VERSION)
BASEIMAGE := ${BASEIMAGE.${OS}}

DOCKERFILE.linux=Dockerfile
DOCKERFILE.windows=Dockerfile.windows
DOCKERFILE := ${DOCKERFILE.${OS}}

TEMP_DIR:=$(shell mktemp -d -t conformanceXXXXXX)

all: build

build:

ifndef VERSION
    $(error VERSION is undefined)
endif
	cp -r ./* ${TEMP_DIR}

	cp ${GINKGO_BIN} ${TEMP_DIR}
	cp ${KUBECTL_BIN} ${TEMP_DIR}
	cp ${E2E_TEST_BIN} ${TEMP_DIR}
	cp ${E2E_GO_RUNNER_BIN} ${TEMP_DIR}/$(GORUNNER)
	cp -r ${CLUSTER_DIR} ${TEMP_DIR}/cluster

	chmod a+rx ${TEMP_DIR}/$(GINKGO)
	chmod a+rx ${TEMP_DIR}/$(KUBECTL)
	chmod a+rx ${TEMP_DIR}/$(E2E_TEST)
	chmod a+rx ${TEMP_DIR}/$(GORUNNER)

	DOCKER_CLI_EXPERIMENTAL=enabled docker buildx build \
		--output=type=${OUTPUT_TYPE} \
		--platform ${OS}/${ARCH} \
		--pull \
		-t ${REGISTRY}/conformance-${IMAGE_NAME_EXTENSION}:${VERSION} \
		--build-arg BASEIMAGE=$(BASEIMAGE) \
		-f ${TEMP_DIR}/${DOCKERFILE} \
		${TEMP_DIR}
	rm -rf "${TEMP_DIR}"

push: build
ifeq ($(OS)/$(ARCH),linux/amd64)
	docker rmi ${REGISTRY}/conformance:${VERSION} 2>/dev/null || true
	docker tag ${REGISTRY}/conformance-${IMAGE_NAME_EXTENSION}:${VERSION} ${REGISTRY}/conformance:${VERSION}
	docker push ${REGISTRY}/conformance:${VERSION}
endif

.PHONY: build push all
