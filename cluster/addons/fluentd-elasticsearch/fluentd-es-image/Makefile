.PHONY:	build push

IMAGE = fluentd-elasticsearch
# By default, pushes to a githash.  To override:
# make push TAG=<version>
# Next TAG=1.6

TAG = $(shell git log -1 --format="format:%H")

build:	
	docker build -t gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	
	gcloud preview docker push gcr.io/google_containers/$(IMAGE):$(TAG)
