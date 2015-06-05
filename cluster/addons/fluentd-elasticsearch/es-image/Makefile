.PHONY:	elasticsearch_logging_discovery build push

# By default, pushes to a githash.  To override:
# make push TAG=<version>
# Next TAG=1.4

TAG = $(shell git log -1 --format="format:%H")

build:	elasticsearch_logging_discovery
	docker build -t gcr.io/google_containers/elasticsearch:$(TAG) .

push:
	gcloud preview docker push gcr.io/google_containers/elasticsearch:$(TAG)

elasticsearch_logging_discovery:
	go build elasticsearch_logging_discovery.go

clean:
	rm elasticsearch_logging_discovery
