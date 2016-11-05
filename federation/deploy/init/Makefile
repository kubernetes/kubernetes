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

IMAGE_NAME?=gcr.kubernetes.io/federation/init
IMAGE_VERSION?=v0.9.0

.PHONY: all
all: build push

.PHONY: build
build:
	# Setting CGO_ENABLED=0 builds a static binary
	CGO_ENABLED=0 go build init.go
	docker build \
	  -t $(IMAGE_NAME):$(IMAGE_VERSION) .
	rm init


.PHONY: push
push:
	gcloud docker push $(IMAGE_NAME):$(IMAGE_VERSION)

