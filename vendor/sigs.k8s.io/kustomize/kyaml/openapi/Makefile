# Copyright 2020 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0

MYGOBIN = $(shell go env GOBIN)
ifeq ($(MYGOBIN),)
MYGOBIN = $(shell go env GOPATH)/bin
endif
KIND_VERSION := "v0.11.1"
API_VERSION ?= "v1.21.2"

.PHONY: all
all: \
	kustomizationapi/swagger.go \
	kubernetesapi/swagger.go \
	kubernetesapi/openapiinfo.go

.PHONY: clean
clean:
	rm kustomizationapi/swagger.go
	rm kubernetesapi/openapiinfo.go

# This will remove all currently built-in schema,
# so think twice before deleting.
# To replace what this will delete typically requires the ability
# to contact a live kubernetes API server.
.PHONY: nuke
nuke: clean
	rm -r kubernetesapi/*

$(MYGOBIN)/go-bindata:
	go install github.com/go-bindata/go-bindata/v3/go-bindata@latest

$(MYGOBIN)/kind:
	( \
		set -e; \
		d=$(shell mktemp -d); cd $$d; \
		wget -O ./kind https://github.com/kubernetes-sigs/kind/releases/download/$(KIND_VERSION)/kind-$(shell uname)-amd64; \
		chmod +x ./kind; \
		mv ./kind $(MYGOBIN); \
		rm -rf $$d; \
	)

.PHONY: kubernetesapi/openapiinfo.go
kubernetesapi/openapiinfo.go:
	./scripts/makeOpenApiInfoDotGo.sh

kustomizationapi/swagger.go: $(MYGOBIN)/go-bindata kustomizationapi/swagger.json
	$(MYGOBIN)/go-bindata \
		--pkg kustomizationapi \
		-o kustomizationapi/swagger.go \
		kustomizationapi/swagger.json

.PHONY: kubernetesapi/swagger.pb
kubernetesapi/swagger.pb: $(MYGOBIN)/kind $(MYGOBIN)/kustomize
	./scripts/fetchSchemaFromCluster.sh $(API_VERSION)

.PHONY: kubernetesapi/swagger.go
kubernetesapi/swagger.go: $(MYGOBIN)/go-bindata kubernetesapi/swagger.pb
	./scripts/generateSwaggerDotGo.sh $(API_VERSION)

$(MYGOBIN)/kustomize:
	$(shell cd ../.. && MYGOBIN=$(MYGOBIN) make $(MYGOBIN)/kustomize)
