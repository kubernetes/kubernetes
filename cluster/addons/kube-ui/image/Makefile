# Makefile for the Docker image gcr.io/google_containers/kube-ui
# MAINTAINER: Tim St. Clair <stclair@google.com>
# If you update this image please check the tag value before pushing.

.PHONY: all container push clean

# Keep this at dev, so no one accidentally blows away the latest published version.
TAG = dev # current version: v1.1
PREFIX = gcr.io/google_containers

all: push

kube-ui: kube-ui.go
	CGO_ENABLED=0 GOOS=linux godep go build -a -installsuffix cgo -ldflags '-w' ./kube-ui.go

container: kube-ui
	docker build -t $(PREFIX)/kube-ui:$(TAG) .

push: container
	gcloud docker push $(PREFIX)/kube-ui:$(TAG)

clean:
	rm -f kube-ui
