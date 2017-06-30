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
#   [ARCH=amd64] [REGISTRY="gcr.io/google-containers"] make (build|push) VERSION={some_released_version_of_kubernetes}

REGISTRY?=gcr.io/google-containers
ARCH?=amd64
HYPERKUBE_BIN?=_output/dockerized/bin/linux/$(ARCH)/hyperkube

BASEIMAGE=gcr.io/google-containers/debian-hyperkube-base-$(ARCH):0.1
TEMP_DIR:=$(shell mktemp -d -t hyperkubeXXXXXX)

all: build

build:

ifndef VERSION
    $(error VERSION is undefined)
endif
	cp -r ./* ${TEMP_DIR}
	cp ../../../${HYPERKUBE_BIN} ${TEMP_DIR}

	chmod a+rx ${TEMP_DIR}/hyperkube

	cd ${TEMP_DIR} && sed -i.back "s|BASEIMAGE|${BASEIMAGE}|g" Dockerfile

	docker build --pull -t ${REGISTRY}/hyperkube-${ARCH}:${VERSION} ${TEMP_DIR}
	rm -rf "${TEMP_DIR}"

push: build
	gcloud docker -- push ${REGISTRY}/hyperkube-${ARCH}:${VERSION}
ifeq ($(ARCH),amd64)
	docker rmi ${REGISTRY}/hyperkube:${VERSION} 2>/dev/null || true
	docker tag ${REGISTRY}/hyperkube-${ARCH}:${VERSION} ${REGISTRY}/hyperkube:${VERSION}
	gcloud docker -- push ${REGISTRY}/hyperkube:${VERSION}
endif

.PHONY: build push all
