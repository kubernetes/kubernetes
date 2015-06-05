# The build rule builds a Docker image that logs all Docker contains logs to
# Google Compute Platform using the Cloud Logging API. The push rule pushes
# the image to DockerHub.
# Satnam Singh (satnam@google.com)

.PHONY:	build push

# By default, pushes to a githash.  To override:
# make push TAG=<version>
# Next TAG=1.7

TAG = $(shell git log -1 --format="format:%H")

build:
	docker build -t gcr.io/google_containers/fluentd-gcp:$(TAG) .

push:
	gcloud preview docker push gcr.io/google_containers/fluentd-gcp:$(TAG)

