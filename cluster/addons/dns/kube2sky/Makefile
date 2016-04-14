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

# Makefile for the Docker image gcr.io/google_containers/kube2sky
# MAINTAINER: Tim Hockin <thockin@google.com>
# If you update this image please bump the tag value before pushing.
#
# Usage:
#   [ARCH=amd64] [TAG=1.14] [REGISTRY=gcr.io/google_containers] [BASEIMAGE=busybox] make (build|push)

# Default registry, arch and tag. This can be overwritten by arguments to make
ARCH?=amd64
TAG?=1.15
REGISTRY?=gcr.io/google_containers
GOLANG_VERSION=1.6
GOARM=6
KUBE_ROOT=$(shell pwd)/../../../..
TEMP_DIR:=$(shell mktemp -d)

ifeq ($(ARCH),amd64)
	BASEIMAGE?=busybox
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


all: container

kube2sky: kube2sky.go
	# Only build kube2sky. This requires go and godep in PATH
	CGO_ENABLED=0 GOARCH=$(ARCH) GOARM=$(GOARM) go build -a -installsuffix cgo --ldflags '-w' ./kube2sky.go

container:
	# Copy the content in this dir to the temp dir
	cp ./* $(TEMP_DIR)

	# Build the binary dockerized. Mount the whole Kubernetes source first, and then the temporary dir to kube2sky source.
	# It runs "make kube2sky" inside the docker container, and the binary is put in the temporary dir.
	docker run -it \
		-v $(KUBE_ROOT):/go/src/k8s.io/kubernetes \
		-v $(TEMP_DIR):/go/src/k8s.io/kubernetes/cluster/addons/dns/kube2sky \
		golang:$(GOLANG_VERSION) /bin/bash -c \
		"go get github.com/tools/godep \
		&& make -C /go/src/k8s.io/kubernetes/cluster/addons/dns/kube2sky kube2sky ARCH=$(ARCH)"

	# Replace BASEIMAGE with the real base image
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

	# And build the image
	docker build -t $(REGISTRY)/kube2sky-$(ARCH):$(TAG) $(TEMP_DIR)

push: container
	gcloud docker push $(REGISTRY)/kube2sky-$(ARCH):$(TAG)

ifeq ($(ARCH),amd64)
	# Backward compatability. TODO: deprecate this image tag
	docker tag -f $(REGISTRY)/kube2sky-$(ARCH):$(TAG) $(REGISTRY)/kube2sky:$(TAG)
	gcloud docker push $(REGISTRY)/kube2sky:$(TAG)
endif

clean:
	rm -f kube2sky

test: clean
	go test -v --vmodule=*=4


.PHONY: all kube2sky container push clean test
