# Copyright 2016 The Kubernetes Authors. All rights reserved.

# Procedure for change:
# 1. Bump the tag number.
# 2. Create an image for the kubernetes user on Dockerhub and push there: make build push
# 3. Temporarily adjust any config files etc. that reference gcr.io/google_containers/pod-resizer:$(TAG)
#    to use kubernetes/fluentd-gcp:$(TAG) instead.
# 4. Test etc.
# 5. Issue PR.
# 6. When PR is approved make the gcr.io version of the image: make build push
# 7. Revert the referendes to kubernetes/fluentd-gcp:$(TAG) to gcr.io/google_containers/pod-resizer:$(TAG) 

OUT_DIR = build
PACKAGE = k8s.io/kubernetes/cluster/addons/pod-resizer
PROJECT = gcr.io/google_containers
TAG = 1.0

# Rules for building the real image for deployment to gcr.io

deps:
		go get -u github.com/tools/godep

compile: nanny/ deps
		GOOS=linux GOARCH=amd64 CGO_ENABLED=0 godep go build -a -o $(OUT_DIR)/pod_nanny nanny/main/pod_nanny.go

test: nanny/
		godep go test ${PACKAGE}/nanny -v

go: compile test

clean:
	rm -rf build

build: go
	docker build -t ${PROJECT}/pod-resizer:$(TAG) .

docker: build

push: docker
	gcloud docker push ${PROJECT}/pod-resizer:$(TAG)
