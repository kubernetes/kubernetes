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

# The build rule builds a Docker image that logs all Docker contains logs to
# Google Compute Platform using the Cloud Logging API. The push rule pushes
# the image to DockerHub.

# Procedure for change:
# 1. Bump the tag number.
# 2. Push to the private repo and test using newer version
# 3. Issue PR.
# 4. When PR is approved make the gcr.io version of the image: make build push
# 5. Issue PR with config files changes

.PHONY:	build push

PREFIX=gcr.io/google_containers
TAG = 1.30

build: build_binaries
	sudo docker build -t $(PREFIX)/fluentd-gcp:$(TAG) .

push:
	docker -- push $(PREFIX)/fluentd-gcp:$(TAG)

build_binaries: clean build_build_container
	docker run -ti -v $(shell pwd)/_output:/_output $(PREFIX)/fluentd-gcp-build:$(TAG)

build_build_container:
	docker build -t $(PREFIX)/fluentd-gcp-build:$(TAG) ./build

clean:
	sudo rm -rf _output
