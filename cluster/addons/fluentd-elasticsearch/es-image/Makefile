.PHONY:	build push

TAG = 1.1

build:	
	docker build -t gcr.io/google_containers/elasticsearch:$(TAG) .

push:	
	gcloud preview docker push gcr.io/google_containers/elasticsearch:$(TAG)
