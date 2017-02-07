# Copyright 2017 The Kubernetes Authors.
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

.PHONY: all build build-image push-image clean

all: build

# The project to which image is to be pushed.
PROJ?=gcr.io/google_containers

# Tag to be used for the image.
TAG?=test

# Name of the etcd-monitor container image.
IMAGE:=$(PROJ)/etcd_version_monitor:$(TAG)

# Build the binary for etcd-monitor.
build: etcd-version-monitor.go
	GOOS=linux CGO_ENABLED=0 go build -o ./bin/etcd-version-monitor etcd-version-monitor.go

# Build the docker image for etcd-monitor
build-image: build
	docker build -t $(IMAGE) .

# Push the docker image to the gcloud image registry.
push-image: build-image
	gcloud docker push $(IMAGE)

# Remove the binary.
clean:
	rm -f bin/etcd-version-monitor

