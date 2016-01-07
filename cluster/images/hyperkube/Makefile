# build the hyperkube image.

VERSION=v1.1.2
ARCH=amd64
BASEIMAGE=debian:jessie
TEMP_DIR:=$(shell mktemp -d)

## Comment in for arm builds, must be run on an arm machine
# ARCH=arm
# need to escape '/' for the regexp below
# BASEIMAGE=armbuild\\/debian:jessie

all: build

build:
	cp ./* ${TEMP_DIR}
	cp ../../saltbase/salt/helpers/safe_format_and_mount ${TEMP_DIR}
	cp ../../saltbase/salt/generate-cert/make-ca-cert.sh ${TEMP_DIR}
	cp ../../../_output/dockerized/bin/linux/${ARCH}/hyperkube ${TEMP_DIR}
	cd ${TEMP_DIR} && sed -i "s/VERSION/${VERSION}/g" master-multi.json master.json kube-proxy.json
	cd ${TEMP_DIR} && sed -i "s/ARCH/${ARCH}/g" master-multi.json master.json kube-proxy.json
	cd ${TEMP_DIR} && sed -i "s/BASEIMAGE/${BASEIMAGE}/g" Dockerfile
	docker build -t gcr.io/google_containers/hyperkube-${ARCH}:${VERSION} ${TEMP_DIR}
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
