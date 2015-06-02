.PHONY:	elasticsearch_logging_discovery build push

# Keep this one version ahead to help prevent accidental pushes.
TAG = 1.4

build:	elasticsearch_logging_discovery
	docker build -t gcr.io/google_containers/elasticsearch:$(TAG) .

push:
	gcloud preview docker push gcr.io/google_containers/elasticsearch:$(TAG)

elasticsearch_logging_discovery:
	go build elasticsearch_logging_discovery.go

clean:
	rm elasticsearch_logging_discovery
