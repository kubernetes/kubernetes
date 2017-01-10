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

# Build the etcd image
#
# Usage:
# 	[TAGS=2.2.1 2.3.7 3.0.14] [REGISTRY=gcr.io/google_containers] [ARCH=amd64] [BASEIMAGE=busybox] make (build|push)

# The image contains different etcd versions to simplify
# upgrades. Thus be careful when removing any tag from here.
#
# NOTE: The etcd upgrade rules are that you can upgrade only 1 minor
# version at a time, and patch release don't matter.
#
# Except from etcd-$(tag) and etcdctl-$(tag) binaries, we also
# need etcd and etcdctl binaries for backward compatibility reasons.
# That binary will be set to the last tag from $(TAGS).
TAGS?=2.2.1 2.3.7 3.0.14
REGISTRY_TAG?=3.0.14
ARCH?=amd64
REGISTRY?=gcr.io/google_containers
GOLANG_VERSION?=1.6.3
GOARM=6
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
ifeq ($(ARCH),s390x)
	BASEIMAGE?=s390x/busybox
endif

build:
	# Copy the content in this dir to the temp dir,
	# without copying the subdirectories.
	find ./ -maxdepth 1 -type f | xargs cp -t $(TEMP_DIR)

	# Compile attachlease
	docker run -it -v $(shell pwd)/../../../:/go/src/k8s.io/kubernetes -v $(TEMP_DIR):/build -e GOARCH=$(ARCH) golang:$(GOLANG_VERSION) \
		/bin/bash -c "CGO_ENABLED=0 go build -o /build/attachlease k8s.io/kubernetes/cluster/images/etcd/attachlease"
	# Compile rollback
	docker run -it -v $(shell pwd)/../../../:/go/src/k8s.io/kubernetes -v $(TEMP_DIR):/build -e GOARCH=$(ARCH) golang:$(GOLANG_VERSION) \
		/bin/bash -c "CGO_ENABLED=0 go build -o /build/rollback k8s.io/kubernetes/cluster/images/etcd/rollback"


ifeq ($(ARCH),amd64)

	# Do not compile if we should make an image for amd64, use the official etcd binaries instead
	# For each release create a tmp dir 'etcd_release_tmp_dir' and unpack the release tar there.
	for tag in $(TAGS); do \
		etcd_release_tmp_dir=$(shell mktemp -d); \
		curl -sSL --retry 5 https://github.com/coreos/etcd/releases/download/v$$tag/etcd-v$$tag-linux-amd64.tar.gz | tar -xz -C $$etcd_release_tmp_dir --strip-components=1; \
		cp $$etcd_release_tmp_dir/etcd $$etcd_release_tmp_dir/etcdctl $(TEMP_DIR)/; \
		cp $(TEMP_DIR)/etcd $(TEMP_DIR)/etcd-$$tag; \
		cp $(TEMP_DIR)/etcdctl $(TEMP_DIR)/etcdctl-$$tag; \
	done
else

	# Download etcd in a golang container and cross-compile it statically
	# For each release create a tmp dir 'etcd_release_tmp_dir' and unpack the release tar there.
	for tag in $(TAGS); do \
		etcd_release_tmp_dir=$(shell mktemp -d); \
		docker run -it -v $$etcd_release_tmp_dir:/etcdbin golang:$(GOLANG_VERSION) /bin/bash -c \
			"git clone https://github.com/coreos/etcd /go/src/github.com/coreos/etcd \
			&& cd /go/src/github.com/coreos/etcd \
			&& git checkout v$$tag \
			&& GOARM=$(GOARM) GOARCH=$(ARCH) ./build \
			&& cp -f bin/$(ARCH)/etcd* bin/etcd* /etcdbin; echo 'done'"; \
		cp $$etcd_release_tmp_dir/etcd $$etcd_release_tmp_dir/etcdctl $(TEMP_DIR)/; \
		cp $(TEMP_DIR)/etcd $(TEMP_DIR)/etcd-$$tag; \
		cp $(TEMP_DIR)/etcdctl $(TEMP_DIR)/etcdctl-$$tag; \
	done

	# Add this ENV variable in order to workaround an unsupported arch blocker
	# The multiarch feature is in an limited and experimental state right now, and etcd should work fine on arm64
	# On arm (which is 32-bit), it can't handle >1GB data in-memory, but it is very unlikely someone tinkering with their limited arm devices would reach such a high usage
	# ppc64le is still quite untested, but compiles and is probably in the process of being validated by IBM.
	cd $(TEMP_DIR) && echo "ENV ETCD_UNSUPPORTED_ARCH=$(ARCH)" >> Dockerfile
endif

	# Replace BASEIMAGE with the real base image
	cd $(TEMP_DIR) && sed -i.bak 's|BASEIMAGE|$(BASEIMAGE)|g' Dockerfile

	# And build the image
	docker build --pull -t $(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) $(TEMP_DIR)

push: build
	gcloud docker -- push $(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG)

ifeq ($(ARCH),amd64)
	# Backward compatibility. TODO: deprecate this image tag
	docker tag $(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) $(REGISTRY)/etcd:$(REGISTRY_TAG)
	gcloud docker -- push $(REGISTRY)/etcd:$(REGISTRY_TAG)
endif

all: build
.PHONY:	build push
