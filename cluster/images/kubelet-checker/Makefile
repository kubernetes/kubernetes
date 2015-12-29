# Build the kubelet-checker image.
#
# Usage:
#   VERSION=v1.1.2 [REGISTRY="gcr.io/google_containers"] make build

REGISTRY?="gcr.io/google_containers"
ARCH=amd64

build:
ifndef VERSION
    $(error VERSION is undefined)
endif
	docker build -t ${REGISTRY}/kubelet-checker-${ARCH}:${VERSION} .

.PHONY: release
