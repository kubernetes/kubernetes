.PHONY:	elasticsearch_logging_discovery build push

# The current value of the tag to be used for building and
# pushing an image to gcr.io
TAG = 1.7

build:	elasticsearch_logging_discovery
	docker build -t gcr.io/google_containers/elasticsearch:$(TAG) .

push:
	gcloud docker push gcr.io/google_containers/elasticsearch:$(TAG)

elasticsearch_logging_discovery:
	go build elasticsearch_logging_discovery.go

clean:
	rm elasticsearch_logging_discovery
