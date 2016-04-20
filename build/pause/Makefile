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

.PHONY:	build push

TAG=2.0
REGISTRY?="gcr.io/google_containers"

# Architectures supported: amd64, arm, arm64 and ppc64le
ARCH?=amd64
GOLANG_VERSION=1.6
TEMP_DIR:=$(shell mktemp -d)

all: push-legacy-too

build:
	cp pause.go Dockerfile $(TEMP_DIR)

	docker run -it -v $(TEMP_DIR):/build golang:$(GOLANG_VERSION) /bin/bash -c \
		"cd /build && CGO_ENABLED=0 GOARM=6 GOARCH=$(ARCH) go build -a -installsuffix cgo -ldflags '-w' ./pause.go"

ifeq ($(ARCH),$(filter $(ARCH),amd64 arm))
# Run goupx for amd64 and arm. The condition above is equal to 'if [[ $ARCH == "amd64" || $ARCH == "arm" ]]' in bash
	docker run -it -v $(TEMP_DIR):/build golang:$(GOLANG_VERSION) /bin/bash -c \
		"cd /build && apt-get update && apt-get install -y upx \
		&& go get github.com/pwaller/goupx && goupx pause"
endif

	# And build the image
	docker build -t $(REGISTRY)/pause-$(ARCH):$(TAG) $(TEMP_DIR)

ifeq ($(ARCH),amd64)
	docker tag -f $(REGISTRY)/pause-$(ARCH):$(TAG) $(REGISTRY)/pause:$(TAG)
endif

push: build
	gcloud docker --server=gcr.io push $(REGISTRY)/pause-$(ARCH):$(TAG)

push-legacy-too: push
ifeq ($(ARCH),amd64)
	gcloud docker push $(REGISTRY)/pause:$(TAG)
endif
