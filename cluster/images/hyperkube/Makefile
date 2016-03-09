# Copyright 2016 The Kubernetes Authors All rights reserved.
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
#   VERSION=v1.1.2 [REGISTRY="gcr.io/google_containers"] make build

REGISTRY?="gcr.io/google_containers"
ARCH=amd64
BASEIMAGE=debian:jessie
TEMP_DIR:=$(shell mktemp -d)

## Comment in for arm builds, must be run on an arm machine
# ARCH=arm
# need to escape '/' for the regexp below
# BASEIMAGE=armbuild\\/debian:jessie

all: build

build:
ifndef VERSION
    $(error VERSION is undefined)
endif
	cp ./* ${TEMP_DIR}
	cp ../../saltbase/salt/helpers/safe_format_and_mount ${TEMP_DIR}
	cp ../../saltbase/salt/generate-cert/make-ca-cert.sh ${TEMP_DIR}
	cp ../../../_output/dockerized/bin/linux/${ARCH}/hyperkube ${TEMP_DIR}
	cd ${TEMP_DIR} && sed -i "s/VERSION/${VERSION}/g" master-multi.json master.json kube-proxy.json
	cd ${TEMP_DIR} && sed -i "s/ARCH/${ARCH}/g" master-multi.json master.json kube-proxy.json
	cd ${TEMP_DIR} && sed -i "s/BASEIMAGE/${BASEIMAGE}/g" Dockerfile
	docker build -t ${REGISTRY}/hyperkube-${ARCH}:${VERSION} ${TEMP_DIR}
	# Backward compatibility.  TODO: deprecate this image tag
ifeq ($(ARCH),amd64)
	docker tag -f ${REGISTRY}/hyperkube-${ARCH}:${VERSION} ${REGISTRY}/hyperkube:${VERSION}
endif

push: build
	gcloud docker push ${REGISTRY}/hyperkube-${ARCH}:${VERSION}
ifeq ($(ARCH),amd64)
	gcloud docker push ${REGISTRY}/hyperkube:${VERSION}
endif

.PHONY: all
