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

ARCH   := amd64
PREFIX ?= gcr.io/google_containers
TAG    ?= 1.0
IMAGE  ?= e2e-net-$(ARCH)

SRCS := $(shell find . -name \*.go)

all: image

net: $(SRCS)
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w'

image: test net
	docker build --pull -t $(PREFIX)/$(IMAGE):$(TAG) .

push: image
	gcloud docker -- push $(PREFIX)/$(IMAGE):$(TAG)

clean:
	rm -f net

test:
	go test ./...

.PHONY: all clean image push test
