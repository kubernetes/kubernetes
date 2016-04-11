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

# Build skydns
#
# Usage:
#   [ARCH=amd64] [TAG=1.0] [REGISTRY=gcr.io/google_containers] [BASEIMAGE=busybox] make (build|push)

# Default registry and arch. This can be overwritten by arguments to make
ARCH?=amd64

# Version of this image, not the version of skydns
TAG=1.0
REGISTRY?=gcr.io/google_containers
GOLANG_VERSION=1.6
GOARM=6
TEMP_DIR:=$(shell mktemp -d)

ifeq ($(ARCH),amd64)
	BASEIMAGE?=busybox
	GOBIN_DIR=/go/bin/
else
	# If not the GOARCH == the host arch, the binary directory is here
	GOBIN_DIR=/go/bin/linux_$(ARCH)
endif

ifeq ($(ARCH),arm)
	BASEIMAGE?=armel/busybox
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=aarch64/busybox
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/busybox
endif

# Do not change this default value
all: skydns

skydns:
	# Only build skydns. This requires go in PATH
	CGO_ENABLED=0 GOARCH=$(ARCH) GOARM=$(GOARM) go get -a -installsuffix cgo --ldflags '-w' github.com/skynetservices/skydns

container:
	# Copy the content in this dir to the temp dir
	cp ./* $(TEMP_DIR)

	# Build skydns in a container. We mount the temporary dir
	docker run -it -v $(TEMP_DIR):$(GOBIN_DIR) golang:$(GOLANG_VERSION) /bin/bash -c "make -C $(GOBIN_DIR) skydns ARCH=$(ARCH)"

	# Replace BASEIMAGE with the real base image
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

	# And build the image
	docker build -t $(REGISTRY)/skydns-$(ARCH):$(TAG) $(TEMP_DIR)

push: container
	gcloud docker push $(REGISTRY)/skydns-$(ARCH):$(TAG)

ifeq ($(ARCH),amd64)
	# Backward compatability. TODO: deprecate this image tag
	docker tag -f $(REGISTRY)/skydns-$(ARCH):$(TAG) $(REGISTRY)/skydns:$(TAG)
	gcloud docker push $(REGISTRY)/skydns:$(TAG)
endif

clean:
	rm -f skydns
