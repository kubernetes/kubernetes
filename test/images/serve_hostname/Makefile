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

# Cross-build the serve_hostname image
#
# Usage:
#       [TAG=v1.5] [PREFIX=gcr.io/google_containers] [ARCH=amd64] [BASEIMAGE=busybox] make container

all: serve_hostname

TAG?=v1.5
ARCH?=amd64
PREFIX?=gcr.io/google_containers
TEST_PREFIX=b.gcr.io/k8s_authenticated_test
GOARM=6
TEMP_DIR:=$(shell mktemp -d)
KUBE_CROSS_TAG=$(shell cat ../../../build/build-image/cross/VERSION)

# Set default base image dynamically for each arch
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

serve_hostname: serve_hostname.go
	# Copy the content in this dir to the temp dir
	cp ./* $(TEMP_DIR)

	# Compile the binary inside a container for reliable builds
	docker run -it -v $(TEMP_DIR):/serve_hostname gcr.io/google_containers/kube-cross:$(KUBE_CROSS_TAG) /bin/bash -c \
            "cd /serve_hostname \
            && CGO_ENABLED=0 GOARM=$(GOARM) GOARCH=$(ARCH) go build -a -installsuffix cgo --ldflags '-w' -o serve_hostname ./serve_hostname.go"

container: serve_hostname
	# Set the base image
	cd $(TEMP_DIR) && sed -i.bak 's|BASEIMAGE|$(BASEIMAGE)|g' Dockerfile

	docker build -t $(PREFIX)/serve_hostname-$(ARCH):$(TAG) $(TEMP_DIR)
	if [ -n "$(TEST_PREFIX)" ]; then \
            docker tag -f $(PREFIX)/serve_hostname-$(ARCH):$(TAG) $(TEST_PREFIX)/serve_hostname-$(ARCH):$(TAG); \
	fi

push: container
	gcloud docker push $(PREFIX)/serve_hostname-$(ARCH):$(TAG)
	if [ -n "$(TEST_PREFIX)" ]; then \
            gcloud docker push $(TEST_PREFIX)/serve_hostname-$(ARCH):$(TAG); \
        fi

clean:
	rm -f serve_hostname
