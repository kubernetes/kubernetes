.PHONY: elasticsearch_discovery build push all

TAG = 1.0

build:
	docker build -t kubernetes/elasticsearch:$(TAG) .

push:
	docker push kubernetes/elasticsearch:$(TAG)

elasticsearch_discovery:	
	go build elasticsearch_discovery.go

all:	elasticsearch_discovery build push 
