# build the hyperkube image.

VERSION=v0.16.2

all:
	cp ../../saltbase/salt/helpers/safe_format_and_mount .
	curl -O https://storage.googleapis.com/kubernetes-release/release/${VERSION}/bin/linux/amd64/hyperkube
	docker build -t gcr.io/google_containers/hyperkube:${VERSION} .
	gcloud preview docker push gcr.io/google_containers/hyperkube:${VERSION}

.PHONY: all
