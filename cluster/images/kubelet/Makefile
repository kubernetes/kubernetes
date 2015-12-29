# Build the kubelet image.
#
# Usage:
#   VERSION=v1.1.2 [REGISTRY="gcr.io/google_containers"] make build
#
# TODO: figure out the best base image

REGISTRY?="gcr.io/google_containers"
ARCH=amd64

build:
ifndef VERSION
    $(error VERSION is undefined)
endif
	cp ../../../_output/dockerized/bin/linux/${ARCH}/kubelet .
	docker build -t ${REGISTRY}/kubelet-${ARCH}:${VERSION} .
	rm -f kubelet

clean:
	rm -f kubelet

.PHONY: release
