.PHONY:	build push

TAG = 1.3

build:
	docker build -t gcr.io/google_containers/kibana:$(TAG) .

push:
	gcloud docker push gcr.io/google_containers/kibana:$(TAG)
