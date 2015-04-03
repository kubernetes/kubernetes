# build the hyperkube image.

VERSION=dev

all:
	cp ../../../_output/release-stage/server/linux-amd64/kubernetes/server/bin/hyperkube ./
	docker build -t gcr.io/google_containers/hyperkube:${VERSION} .
	gcloud preview docker push gcr.io/google_containers/hyperkube:${VERSION}

.PHONY: all
