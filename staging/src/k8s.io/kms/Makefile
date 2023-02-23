# The architecture of the image
ARCH ?= amd64

# Output type of docker buildx build
OUTPUT_TYPE ?= type=registry

REGISTRY_NAME ?= localhost
REGISTRY_PORT ?= 5000
REGISTRY ?= $(REGISTRY_NAME):$(REGISTRY_PORT)
KUBERNETES_VERSION ?= v1.26.0
PLUGIN_NAME ?= base64

# build base64 plugin
.PHONY: base64-docker-build
base64-docker-build:
	docker buildx build \
		--no-cache \
		--platform="linux/$(ARCH)" \
		--output=$(OUTPUT_TYPE) \
		-t $(REGISTRY)/base64:e2e  \
		-f plugins/base64/Dockerfile ../ \
		--progress=plain; \

# this target builds the plugin and push it to the local registry and connect it to the kind cluster
# kind cluster is setup using kubetest2 in test-infra job
.PHONY: e2e-connect-registry
e2e-connect-registry:
	./scripts/setup-local-registry.sh $(PLUGIN_NAME)
	./scripts/connect-registry.sh &

.PHONY: setup-local-registry
setup-local-registry:
	./scripts/setup-local-registry.sh $(PLUGIN_NAME)

# this target is used to test e2e locally with kind cluster
.PHONY: e2e-local-setup-kind
e2e-local-setup-kind: setup-local-registry
	./scripts/setup-kind-cluster.sh &
	./scripts/connect-registry.sh &
	sleep 90s
