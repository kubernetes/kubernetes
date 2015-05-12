.PHONY:	elasticsearch_logging_discovery build push

TAG = 1.2

build:	elasticsearch_logging_discovery	
	docker build -t gcr.io/google_containers/elasticsearch:$(TAG) .

push:	
	gcloud preview docker push gcr.io/google_containers/elasticsearch:$(TAG)

elasticsearch_logging_discovery:
	go build elasticsearch_logging_discovery.go
