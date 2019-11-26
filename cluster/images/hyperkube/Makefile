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

# Build the hyperkube image.
#
# Usage:
#   [ARCH=amd64] [REGISTRY="staging-k8s.gcr.io"] make (build|push) VERSION={some_released_version_of_kubernetes}

REGISTRY?=staging-k8s.gcr.io
ARCH?=amd64
OUT_DIR?=_output

BASEIMAGE=k8s.gcr.io/debian-hyperkube-base-$(ARCH):0.12.1

LOCAL_OUTPUT_PATH=$(shell pwd)/../../../$(OUT_DIR)/local/bin/linux/$(ARCH)
DOCKERIZED_OUTPUT_PATH=$(shell pwd)/../../../$(OUT_DIR)/dockerized/bin/linux/$(ARCH)
OUTPUT_PATH?=$(shell test -d $(LOCAL_OUTPUT_PATH) && echo $(LOCAL_OUTPUT_PATH) || echo $(DOCKERIZED_OUTPUT_PATH))

TEMP_DIR:=$(shell mktemp -d -t hyperkubeXXXXXX)

all: build

build:

ifndef VERSION
    $(error VERSION is undefined)
endif
	cp -r ./* ${TEMP_DIR}

	tar -cvzf ${TEMP_DIR}/binaries.tgz  -C ${OUTPUT_PATH} kube-apiserver kube-controller-manager \
		kube-proxy kube-scheduler kubectl kubelet

	chmod a+rx ${TEMP_DIR}/hyperkube

	cd ${TEMP_DIR} && sed -i.back "s|BASEIMAGE|${BASEIMAGE}|g" Dockerfile

	docker build --pull -t ${REGISTRY}/hyperkube-${ARCH}:${VERSION} ${TEMP_DIR}
	rm -rf "${TEMP_DIR}"

push: build
	docker push ${REGISTRY}/hyperkube-${ARCH}:${VERSION}
ifeq ($(ARCH),amd64)
	docker rmi ${REGISTRY}/hyperkube:${VERSION} 2>/dev/null || true
	docker tag ${REGISTRY}/hyperkube-${ARCH}:${VERSION} ${REGISTRY}/hyperkube:${VERSION}
	docker push ${REGISTRY}/hyperkube:${VERSION}
endif

.PHONY: build push all
