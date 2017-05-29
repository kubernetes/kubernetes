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

REGISTRY?=gcr.io/google_containers
REGISTRY_TAG?=1.0.1
ARCH?=amd64
NAME?=no-snat-test-proxy

build:
	go build --ldflags '-linkmode external -extldflags "-static"' -o $(NAME) main.go
	docker build -t $(REGISTRY)/$(NAME)-$(ARCH):$(REGISTRY_TAG) .
	rm $(NAME)

push: build
	gcloud docker -- push $(REGISTRY)/$(NAME)-$(ARCH):$(REGISTRY_TAG)

all: build
.PHONY: build push