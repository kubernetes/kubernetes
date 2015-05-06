# build the kubelet image.

# TODO: figure out the best base image
# TODO: figure out how to best source the nsenter binary, if necessary

VERSION=v0.16

release:
	curl -O https://storage.googleapis.com/kubernetes-release/release/${VERSION}/bin/linux/amd64/kubelet
	docker build -t gcr.io/google_containers/kubelet:${VERSION} .
	gcloud preview docker push gcr.io/google_containers/kubelet:${VERSION}

local:
	docker build -t gcr.io/google_containers/kubelet:${VERSION} .

.PHONY: release
