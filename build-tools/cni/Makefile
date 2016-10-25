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

# Build the CNI binaries.
#
# Usage:
#	[ARCH=amd64] [CNI_RELEASE=v0.3.0] make (build|push)

ARCH?=amd64
CNI_RELEASE?=07a8a28637e97b22eb8dfe710eeae1344f69d16e
CNI_TARBALL=cni-$(ARCH)-$(CNI_RELEASE).tar.gz
CUR_DIR=$(shell pwd)
OUTPUT_DIR=$(CUR_DIR)/output

all: build

build:
	mkdir -p $(OUTPUT_DIR)

	docker run -it -v $(OUTPUT_DIR):/output golang:1.6 /bin/bash -c "\
	git clone https://github.com/containernetworking/cni\
	&& cd cni \
	&& git checkout $(CNI_RELEASE) \
	&& CGO_ENABLED=0 GOOS=linux GOARCH=$(ARCH) ./build \
	&& tar -zcvf $(CNI_TARBALL) bin/ \
	&& mv $(CNI_TARBALL) /output/"

# Backward Compatibility
ifeq ($(ARCH),amd64)
	cp $(OUTPUT_DIR)/$(CNI_TARBALL) $(OUTPUT_DIR)/cni-$(CNI_RELEASE).tar.gz
endif

push: build
	gsutil cp $(OUTPUT_DIR)/$(CNI_TARBALL) gs://kubernetes-release/network-plugins

ifeq ($(ARCH),amd64)
	gsutil cp $(OUTPUT_DIR)/cni-$(CNI_RELEASE).tar.gz gs://kubernetes-release/network-plugins
endif

clean:
	rm -rf output/

.PHONY: all
