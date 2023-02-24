# Copyright 2023 The Kubernetes Authors.
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

# The architecture of the image
ARCH ?= amd64

# Output type of docker buildx build
OUTPUT_TYPE ?= type=registry

REGISTRY_NAME ?= localhost
REGISTRY_PORT ?= 5000
REGISTRY ?= $(REGISTRY_NAME):$(REGISTRY_PORT)
KUBERNETES_VERSION ?= v1.26.0

# build base64 plugin
.PHONY: base64-docker-build
base64-docker-build:
	docker buildx build \
		--no-cache \
		--platform="linux/$(ARCH)" \
		--output=$(OUTPUT_TYPE) \
		-t $(REGISTRY)/base64:e2e  \
		-f Dockerfile ../../../../ \
		--progress=plain; \

# these targets builds the plugin and push it to the local registry and connect it to the kind cluster
# kind cluster is setup using kubetest2 in test-infra job
.PHONY: e2e-connect-registry
e2e-connect-registry:
	./../../../../../../../test/e2e/testing-manifests/auth/encrypt/connect-registry.sh &

.PHONY: setup-local-registry
setup-local-registry:
	./setup-local-registry.sh
