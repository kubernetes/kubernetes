# build the kubelet image.

# TODO: figure out the best base image
# TODO: figure out how to best source the nsenter binary, if necessary
# TODO: figure out how to make a single source of version info for these
#       images
VERSION=v0.16
BIN_DIR=bin/linux/amd64
LOCAL_OUTPUT_DIR=../../../_output/local

release: clean
	curl -O https://storage.googleapis.com/kubernetes-release/release/${VERSION}/${BIN_DIR}/kubelet
	docker build -t gcr.io/google_containers/kubelet:${VERSION} .
	gcloud preview docker push gcr.io/google_containers/kubelet:${VERSION}

local: clean
	cp ${LOCAL_OUTPUT_DIR}/${BIN_DIR}/kubelet .
	docker build -t gcr.io/google_containers/kubelet .

clean:
	rm -f kubelet

.PHONY: release
