# By default, pushes to a githash.  To override:
# make push TAG=<version>

TAG = $(shell git log -1 --format="format:%H")

all:
	@echo "try 'make image' or 'make push'"

image:
	docker build -t gcr.io/google_containers/dnsutils:$(TAG) .

push:
	gcloud preview docker push gcr.io/google_containers/dnsutils:$(TAG)
