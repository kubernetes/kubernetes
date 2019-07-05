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
ARCH?=amd64
OUT_DIR?=_output

LOCAL_OUTPUT_PATH=$(shell pwd)/../../../$(OUT_DIR)/local/bin/linux/$(ARCH)
DOCKERIZED_OUTPUT_PATH=$(shell pwd)/../../../$(OUT_DIR)/dockerized/bin/linux/$(ARCH)

GINKGO_BIN?=$(shell test -f $(LOCAL_OUTPUT_PATH)/ginkgo && echo $(LOCAL_OUTPUT_PATH)/ginkgo || echo $(DOCKERIZED_OUTPUT_PATH)/ginkgo)
KUBECTL_BIN?=$(shell test -f $(LOCAL_OUTPUT_PATH)/kubectl && echo $(LOCAL_OUTPUT_PATH)/kubectl || echo $(DOCKERIZED_OUTPUT_PATH)/kubectl)
E2E_TEST_BIN?=$(shell test -f $(LOCAL_OUTPUT_PATH)/e2e.test && echo $(LOCAL_OUTPUT_PATH)/e2e.test || echo $(DOCKERIZED_OUTPUT_PATH)/e2e.test)
E2E_GO_RUNNER_BIN?=$(shell test -f $(LOCAL_OUTPUT_PATH)/go-runner && echo $(LOCAL_OUTPUT_PATH)/go-runner || echo $(DOCKERIZED_OUTPUT_PATH)/go-runner)

CLUSTER_DIR?=$(shell pwd)/../../../cluster/

BASEIMAGE=k8s.gcr.io/debian-hyperkube-base-$(ARCH):0.12.1
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
	cp ${E2E_GO_RUNNER_BIN} ${TEMP_DIR}/gorunner
	cp -r ${CLUSTER_DIR} ${TEMP_DIR}/cluster

	chmod a+rx ${TEMP_DIR}/ginkgo
	chmod a+rx ${TEMP_DIR}/kubectl
	chmod a+rx ${TEMP_DIR}/e2e.test
	chmod a+rx ${TEMP_DIR}/gorunner

	cd ${TEMP_DIR} && sed -i.back "s|BASEIMAGE|${BASEIMAGE}|g" Dockerfile

	docker build --pull -t ${REGISTRY}/conformance-${ARCH}:${VERSION} ${TEMP_DIR}
	rm -rf "${TEMP_DIR}"

push: build
	docker push ${REGISTRY}/conformance-${ARCH}:${VERSION}
ifeq ($(ARCH),amd64)
	docker rmi ${REGISTRY}/conformance:${VERSION} 2>/dev/null || true
	docker tag ${REGISTRY}/conformance-${ARCH}:${VERSION} ${REGISTRY}/conformance:${VERSION}
	docker push ${REGISTRY}/conformance:${VERSION}
endif

.PHONY: build push all
