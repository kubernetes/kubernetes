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
#   [ARCH=amd64] [REGISTRY="gcr.io/google_containers"] make (build|push) VERSION={some_released_version_of_kubernetes}

REGISTRY?=gcr.io/google_containers
ARCH?=amd64
TEMP_DIR:=$(shell mktemp -d)

all: build

build:
ifndef VERSION
    $(error VERSION is undefined)
endif
	cp -r ./* ${TEMP_DIR}
	cp ../../../_output/dockerized/bin/linux/${ARCH}/kube-discovery ${TEMP_DIR}
	sudo docker build -t ${REGISTRY}/kube-discovery-${ARCH}:${VERSION} ${TEMP_DIR}
	rm -rf "${TEMP_DIR}"

push: build
	gcloud docker push ${REGISTRY}/kube-discovery-${ARCH}:${VERSION}
	docker tag -f ${REGISTRY}/kube-discovery-${ARCH}:${VERSION} ${REGISTRY}/kube-discovery:${VERSION}
	gcloud docker push ${REGISTRY}/kube-discovery:${VERSION}

.PHONY: all

