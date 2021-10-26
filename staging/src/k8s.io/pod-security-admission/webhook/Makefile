# Copyright 2021 The Kubernetes Authors.
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

.PHONY: build container push clean

ENTRYPOINT = "../cmd/webhook/webhook.go"
EXECUTABLE = "pod-security-webhook"

# Relative to location in staging dir
KUBE_ROOT = "../../../../.."

IMAGE_DOCKERFILE = "Dockerfile"
REGISTRY ?= "gcr.io/k8s-staging-sig-auth"
IMAGE ?= "$(REGISTRY)/pod-security-webhook"
TAG ?= "latest"

OS ?= linux
ARCH ?= amd64

# Builds the PodSecurity webhook binary.
build:
	@echo Building PodSecurity webhook...
	@LDFLAGS=`cd -P . && /usr/bin/env bash -c '. $(KUBE_ROOT)/hack/lib/version.sh && KUBE_ROOT=$(KUBE_ROOT) KUBE_GO_PACKAGE=k8s.io/kubernetes kube::version::ldflags'`; \
	GOOS=$(OS) GOARCH=$(ARCH) CGO_ENABLED=0 go build -o $(EXECUTABLE) -ldflags "$$LDFLAGS" $(ENTRYPOINT)
	@echo Done!

# Builds the PodSecurity webhook Docker image.
container: build
	@echo Building PodSecurity webhook image...
	@docker build \
		-f $(IMAGE_DOCKERFILE) \
		-t $(IMAGE):$(TAG) .
	@echo Done!

# Publishes the PodSecurity webhook Docker image to the configured registry.
push:
	@docker push $(IMAGE):$(TAG)

# Removes Pod Security Webhook artifacts.
clean:
	rm $(EXECUTABLE)
