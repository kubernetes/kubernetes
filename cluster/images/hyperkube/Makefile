# build the hyperkube image.

VERSION=v1.1.2
ARCH=amd64
BASEIMAGE=debian:jessie

## Comment in for arm builds, must be run on an arm machine
# ARCH=arm
# need to escape '/' for the regexp below
# BASEIMAGE=armbuild\\/debian:jessie

all: build

build:
	cp ../../saltbase/salt/helpers/safe_format_and_mount .
	cp ../../saltbase/salt/generate-cert/make-ca-cert.sh  .
	curl -O https://storage.googleapis.com/kubernetes-release/release/${VERSION}/bin/linux/${ARCH}/hyperkube
	sed -i "s/VERSION/${VERSION}/g" master-multi.json master.json kube-proxy.json
	sed -i "s/ARCH/${ARCH}/g" master-multi.json master.json kube-proxy.json
	sed -i "s/BASEIMAGE/${BASEIMAGE}/g" Dockerfile
	docker build -t gcr.io/google_containers/hyperkube-${ARCH}:${VERSION} .
	# Backward compatability.  TODO: deprecate this image tag
ifeq ($(ARCH),amd64)
	docker tag -f gcr.io/google_containers/hyperkube-${ARCH}:${VERSION} gcr.io/google_containers/hyperkube:${VERSION}
endif

push: build
	gcloud docker push gcr.io/google_containers/hyperkube-${ARCH}:${VERSION}
ifeq ($(ARCH),amd64)
	gcloud docker push gcr.io/google_containers/hyperkube:${VERSION}
endif

clean:
	rm -f safe_format_and_mount make-ca-cert.sh hyperkube

.PHONY: all
