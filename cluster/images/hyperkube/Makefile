# build the hyperkube image.

VERSION=v1.0.1

all:
	cp ../../saltbase/salt/helpers/safe_format_and_mount .
	curl -O https://storage.googleapis.com/kubernetes-release/release/${VERSION}/bin/linux/amd64/hyperkube
	sed -i "s/VERSION/${VERSION}/g" master-multi.json master.json
	docker build -t gcr.io/google_containers/hyperkube:${VERSION} .
	gcloud docker push gcr.io/google_containers/hyperkube:${VERSION}

.PHONY: all
