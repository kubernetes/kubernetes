# Copyright 2020 The Kubernetes Authors.
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

REGISTRY ?= registry.k8s.io/e2e-test-images
REMOTE_DOCKER_URL ?=
DOCKER_CERT_PATH ?= "$(HOME)/.docker"
export

ALL_IMAGES = powershell-helper

sub-build-%:
	img_version=$(shell cat $*/VERSION); \
	docker --tlsverify --tlscacert "$(DOCKER_CERT_PATH)/ca.pem" \
		--tlscert "$(DOCKER_CERT_PATH)/cert.pem" --tlskey "$(DOCKER_CERT_PATH)/key.pem" \
		-H "$(REMOTE_DOCKER_URL)" build --no-cache --pull -t "$(REGISTRY)/$*:$${img_version}" \
		-f $*/Dockerfile_windows $*/

sub-push-%:
	img_version=`cat $*/VERSION`; \
	docker --tlsverify --tlscacert "$(DOCKER_CERT_PATH)/ca.pem" \
		--tlscert "$(DOCKER_CERT_PATH)/cert.pem" --tlskey "$(DOCKER_CERT_PATH)/key.pem" \
		-H "$(REMOTE_DOCKER_URL)" push "$(REGISTRY)/$*:$${img_version}"

sub-repush-as-linux-%:
	img_version=$(shell cat $*/VERSION); \
	docker buildx build --progress=plain --no-cache --pull --output=type=registry --platform "linux/amd64" \
		--build-arg SOURCE="$(REGISTRY)/$*:$${img_version}" -t "$(REGISTRY)/$*:$${img_version}"-linux-cache $*/

all-build: $(foreach image, ${ALL_IMAGES}, sub-build-${image})

all-push: all-build $(foreach image, ${ALL_IMAGES}, sub-push-${image})

all-push-as-linux: all-push $(foreach image, ${ALL_IMAGES}, sub-repush-as-linux-${image})

.PHONY: all-build all-push all-push-as-linux
