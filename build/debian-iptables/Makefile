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

.PHONY:	build push

REGISTRY?="staging-k8s.gcr.io"
IMAGE=debian-iptables
TAG?=v10.1
ARCH?=amd64
TEMP_DIR:=$(shell mktemp -d)

BASEIMAGE?=k8s.gcr.io/debian-base-$(ARCH):0.3.2

build:
	cp ./* $(TEMP_DIR)
	cd $(TEMP_DIR) && sed -i "s|BASEIMAGE|$(BASEIMAGE)|g" Dockerfile

ifneq ($(ARCH),amd64)
	# Register /usr/bin/qemu-ARCH-static as the handler for non-x86 binaries in the kernel
	docker run --rm --privileged multiarch/qemu-user-static:register --reset
endif

	docker build --pull -t $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG) $(TEMP_DIR)

push: build
	docker push $(REGISTRY)/$(IMAGE)-$(ARCH):$(TAG)

all: push
