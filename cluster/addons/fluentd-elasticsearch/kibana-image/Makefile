.PHONY:	build push

TAG = 1.2

build:
	docker build -t gcr.io/google_containers/kibana:$(TAG) .

push:
	gcloud preview docker push gcr.io/google_containers/kibana:$(TAG)
