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

# build Kubemark image from currently built binaries containing both 'real' master and Hollow Node.
# This makefile assumes that the kubemark binary is present in this directory.

REGISTRY?=gcr.io
PROJECT?=google_containers

all: gcloudpush

build:
	docker build --pull -t $(REGISTRY)/$(PROJECT)/kubemark .

gcloudpush: build
	gcloud docker -- push $(REGISTRY)/$(PROJECT)/kubemark

push: build
	docker -- push $(REGISTRY)/$(PROJECT)/kubemark

.PHONY: all build gcloudpush push
