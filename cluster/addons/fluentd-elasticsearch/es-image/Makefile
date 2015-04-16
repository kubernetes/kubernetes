.PHONY:	build push

TAG = 1.0

build:	
	docker build -t gcr.io/google_containers/elasticsearch:$(TAG) .

push:	
	docker push gcr.io/google_containers/elasticsearch:$(TAG)
