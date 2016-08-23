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
#       [TAG=v1.4] [PREFIX=gcr.io/google_containers] [ARCH=amd64] make container

all: serve_hostname

TAG?=v1.4
ARCH?=amd64
PREFIX?=gcr.io/google_containers
TEST_PREFIX=b.gcr.io/k8s_authenticated_test
GOARM=6
TEMP_DIR:=$(shell mktemp -d)
KUBE_CROSS_TAG=$(shell cat ../../../build/build-image/cross/VERSION)

ifeq ($(ARCH),amd64)
        CC=x86_64-linux-gnu-gcc
endif
ifeq ($(ARCH),arm)
        CC=arm-linux-gnueabi-gcc
endif
ifeq ($(ARCH),arm64)
        CC=aarch64-linux-gnu-gcc
endif
ifeq ($(ARCH),ppc64le)
        CC=powerpc64le-linux-gnu-gcc
endif

serve_hostname: serve_hostname.go
	# Copy the content in this dir to the temp dir
	cp ./* $(TEMP_DIR)

	docker run -it -v $(TEMP_DIR):/serve_hostname gcr.io/google_containers/kube-cross:$(KUBE_CROSS_TAG) /bin/bash -c \
	    "cd /serve_hostname \
	    && CGO_ENABLED=0 GOARM=$(GOARM) GOARCH=$(ARCH) CC=$(CC) CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' /serve_hostname/serve_hostname.go"


container: serve_hostname
	docker build -t $(PREFIX)/serve_hostname:$(TAG) $(TEMP_DIR)
	if [ -n "$(TEST_PREFIX)" ]; then \
	    docker tag -f $(PREFIX)/serve_hostname:$(TAG) $(TEST_PREFIX)/serve_hostname:$(TAG); \
	fi

push:
	gcloud docker push $(PREFIX)/serve_hostname:$(TAG)
	if [ -n "$(TEST_PREFIX)" ]; then \
	    gcloud docker push $(TEST_PREFIX)/serve_hostname:$(TAG); \
	fi

clean:
	rm -f serve_hostname
