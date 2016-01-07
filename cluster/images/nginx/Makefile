### Nginx image
# This image is used to run nginx on the master.

# Current 'live' version is v1
# Please make sure that you update this comment if you build/push new
# versions.
VERSION=v2

all:
	docker build -t gcr.io/google_containers/nginx:${VERSION} .
	gcloud docker push gcr.io/google_containers/nginx:${VERSION}

.PHONY: all

