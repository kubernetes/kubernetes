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

BUILD_DIR=/tmp/hyperkube
CNI_RELEASE=07a8a28637e97b22eb8dfe710eeae1344f69d16e

UNAME_S:=$(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SED_CMD?=sed -i ""
endif
ifeq ($(UNAME_S),Linux)
	SED_CMD?=sed -i
endif

ifeq ($(ARCH),amd64)
	BASEIMAGE?=debian:jessie
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=armel/debian:jessie
	QEMUARCH=arm
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=aarch64/debian:jessie
	QEMUARCH=aarch64
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/debian:jessie
	QEMUARCH=ppc64le
endif

all: build

build:

ifndef VERSION
    $(error VERSION is undefined)
endif

ifeq (,$(wildcard ../../../_output/dockerized/bin/))
    $(error Kubernetes binaries must be build. Please run 'make quick-release' in root directory.)
endif

	# Inital folder layout
	rm -rf ${BUILD_DIR}/*
	mkdir -p ${BUILD_DIR}/cni-bin \
           ${BUILD_DIR}/addons/singlenode ${BUILD_DIR}/addons/multinode \
           ${BUILD_DIR}/static-pods/singlenode ${BUILD_DIR}/static-pods/multinode


 	# Binaries and scripts
	cp -r ./Dockerfile ./setup-files.sh ./copy-addons.sh ./cni-conf ${BUILD_DIR}
	cp ../../saltbase/salt/generate-cert/make-ca-cert.sh ${BUILD_DIR}
	cp ../../../_output/dockerized/bin/linux/${ARCH}/hyperkube ${BUILD_DIR}
 	# Make scripts executable before they are copied into the Docker image. If we make them executable later, in another layer
 	# they'll take up twice the space because the new executable binary differs from the old one, but everything is cached in layers.
	cd ${BUILD_DIR} && chmod a+rx \
	  hyperkube \
	  setup-files.sh \
	  make-ca-cert.sh \
	  copy-addons.sh
	# Download CNI
	curl -sSL --retry 5 https://storage.googleapis.com/kubernetes-release/network-plugins/cni-${ARCH}-${CNI_RELEASE}.tar.gz | tar -xz -C ${BUILD_DIR}/cni-bin


	# Manifests
	cp ./static-pods/master.json \
	   ./static-pods/etcd.json \
	   ./static-pods/addon-manager-singlenode.json \
	   ./static-pods/kube-proxy.json \
	      ${BUILD_DIR}/static-pods/singlenode
	# Manifests for multinode
	cp ./static-pods/master-multi.json \
	   ./static-pods/addon-manager-multinode.json \
	      ${BUILD_DIR}/static-pods/multinode


	# Addons
	cp ../../addons/dns/skydns-rc.yaml.base ${BUILD_DIR}/addons/singlenode/skydns-rc.yaml
	cp ../../addons/dns/skydns-svc.yaml.base ${BUILD_DIR}/addons/singlenode/skydns-svc.yaml
	cp ../../addons/dashboard/dashboard-controller.yaml ${BUILD_DIR}/addons/singlenode/
	cp ../../addons/dashboard/dashboard-service.yaml ${BUILD_DIR}/addons/singlenode/
	# Addons for multinode; all singlenode addons plus kube-proxy (and soon flannel)
	cp ${BUILD_DIR}/addons/singlenode/*.yaml ${BUILD_DIR}/addons/multinode/
	cp kube-proxy-ds.yaml ${BUILD_DIR}/addons/multinode/kube-proxy.yaml


	# Fill placeholders in templates
	cd ${BUILD_DIR} && sed -i.back "s|VERSION|${VERSION}|g" addons/*/*.yaml static-pods/*/*.json
	cd ${BUILD_DIR} && sed -i.back "s|REGISTRY|${REGISTRY}|g" addons/*/*.yaml static-pods/*/*.json
	cd ${BUILD_DIR} && sed -i.back "s|ARCH|${ARCH}|g" addons/*/*.yaml  static-pods/*/*.json
	cd ${BUILD_DIR} && sed -i.back "s|ARCH|${QEMUARCH}|g" Dockerfile
	cd ${BUILD_DIR} && sed -i.back "s|BASEIMAGE|${BASEIMAGE}|g" Dockerfile
	cd ${BUILD_DIR} && sed -i.back "s|-amd64|-${ARCH}|g" addons/*/*.yaml
	cd ${BUILD_DIR} && sed -i.back "s|__PILLAR__DNS__REPLICAS__|1|g;s|__PILLAR__DNS__SERVER__|10.0.0.10|g;" addons/*/skydns*.yaml
	cd ${BUILD_DIR} && sed -i.back "s|__PILLAR__DNS__DOMAIN__|cluster.local|g;s|__PILLAR__FEDERATIONS__DOMAIN__MAP__||g;" addons/*/skydns*.yaml
	cd ${BUILD_DIR} && rm -f addons/*/*.back static-pods/*/*.back Dockerfile.back


	# Build hyperkube image
ifeq ($(ARCH),amd64)
	# When building "normally" for amd64, remove the whole line, it has no part in the amd64 image
	cd ${BUILD_DIR} && ${SED_CMD} "/CROSS_BUILD_/d" Dockerfile
else
	cd ${BUILD_DIR} && ${SED_CMD} "s/CROSS_BUILD_//g" Dockerfile
	# When cross-building, only the placeholder "CROSS_BUILD_" should be removed
	# Register /usr/bin/qemu-ARCH-static as the handler for ARM binaries in the kernel
	docker run --rm --privileged multiarch/qemu-user-static:register --reset
	curl -sSL --retry 5 https://github.com/multiarch/qemu-user-static/releases/download/v2.5.0/x86_64_qemu-${QEMUARCH}-static.tar.xz | tar -xJ -C ${BUILD_DIR}
endif
	docker build -t ${REGISTRY}/hyperkube-${ARCH}:${VERSION} ${BUILD_DIR}


	# Push to registry
push: build
	gcloud docker push ${REGISTRY}/hyperkube-${ARCH}:${VERSION}
ifeq ($(ARCH),amd64)
	docker tag -f ${REGISTRY}/hyperkube-${ARCH}:${VERSION} ${REGISTRY}/hyperkube:${VERSION}
	gcloud docker push ${REGISTRY}/hyperkube:${VERSION}
endif

.PHONY: all
