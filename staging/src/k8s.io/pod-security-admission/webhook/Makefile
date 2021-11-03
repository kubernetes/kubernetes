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
	GOOS=$(OS) GOARCH=$(ARCH) CGO_ENABLED=0 go build -o $(EXECUTABLE) -trimpath -ldflags "$$LDFLAGS" $(ENTRYPOINT)
	@echo Done!

# Builds the PodSecurity webhook Docker image.
container: build
	@echo Building PodSecurity webhook image...
	@docker build \
		-f $(IMAGE_DOCKERFILE) \
		-t $(IMAGE):$(TAG) .
	@echo Done!

# Creates a CA and serving certificate valid for webhook.pod-security-webhook.svc
certs:
	rm -fr pki
	mkdir -p pki
	openssl genrsa -out pki/ca.key 2048
	openssl req -new -x509 -days 3650 -key pki/ca.key -subj "/CN=pod-security-webhook-ca-$(date +%s)" -out pki/ca.crt

	openssl req -newkey rsa:2048 -nodes -keyout pki/tls.key -subj "/CN=webhook.pod-security-webhook.svc" -out pki/tls.csr

	echo "subjectAltName=DNS:webhook.pod-security-webhook.svc" > pki/extensions.txt
	echo "extendedKeyUsage=serverAuth" >> pki/extensions.txt
	openssl x509 -req -extfile pki/extensions.txt -days 730 -in pki/tls.csr -CA pki/ca.crt -CAkey pki/ca.key -CAcreateserial -out pki/tls.crt

# Publishes the PodSecurity webhook Docker image to the configured registry.
push:
	@docker push $(IMAGE):$(TAG)

# Removes Pod Security Webhook artifacts.
clean:
	rm -f $(EXECUTABLE)
	rm -fr pki
