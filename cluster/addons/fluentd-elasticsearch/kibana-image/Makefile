.PHONY:	build push

# By default, pushes to a githash.  To override:
# make push TAG=<version>
# Next TAG=1.4

TAG = $(shell git log -1 --format="format:%H")

build:
	docker build -t gcr.io/google_containers/kibana:$(TAG) .

push:
	gcloud preview docker push gcr.io/google_containers/kibana:$(TAG)
