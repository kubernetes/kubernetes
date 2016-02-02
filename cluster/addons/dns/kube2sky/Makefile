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

.PHONY: all kube2sky container push clean test

TAG = 1.14
PREFIX = gcr.io/google_containers

all: container

kube2sky: kube2sky.go
	GOOS=linux CGO_ENABLED=0 godep go build -a -installsuffix cgo --ldflags '-w' ./kube2sky.go

container: kube2sky
	docker build -t $(PREFIX)/kube2sky:$(TAG) .

push:
	gcloud docker push $(PREFIX)/kube2sky:$(TAG)

clean:
	rm -f kube2sky

test: clean
	godep go test -v --vmodule=*=4
