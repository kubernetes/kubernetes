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

OS ?= linux
ARCH ?= amd64
TARGET ?= $(CURDIR)
GOARM = 7
GOLANG_VERSION ?= latest
SRC_DIR = $(notdir $(shell pwd))
KUBE_CROSS_VERSION ?= $(shell cat ../../../build/build-image/cross/VERSION)
export

# Build v1.32.1 to ensure the current release supports a prior version of the sample apiserver
# Get without building to populate module cache
# Then, get with OS/ARCH-specific env to build
bin:
	docker run --rm -i -v "${TARGET}:${TARGET}:Z" registry.k8s.io/build-image/kube-cross:${KUBE_CROSS_VERSION} \
		/bin/bash -c "\
			mkdir -p /go/src /go/bin && \
			git clone https://github.com/kubernetes/sample-apiserver /go/src/k8s.io/sample-apiserver && \
			cd /go/src/k8s.io/sample-apiserver && \
			git checkout tags/v0.32.1 && \
			go mod tidy && \
			GO111MODULE=on CGO_ENABLED=0 GOARM=${GOARM} GOOS=${OS} GOARCH=${ARCH} go install . && \
			find /go/bin -name sample-apiserver* -exec cp {} ${TARGET}/sample-apiserver \;"
			# for arm, go install uses /go/bin/linux_arm, so just find the file and copy it to the
			# root so we can copy it out from this throw away container image from a standard location.
			# Windows executables have .exe extension, which is why we're searching sample-apiserver*

.PHONY: bin
