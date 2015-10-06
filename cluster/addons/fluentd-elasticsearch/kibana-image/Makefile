.PHONY:	build push

TAG = 1.3

build:
	docker build -t beta.gcr.io/google_containers/kibana:$(TAG) .

push:
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/kibana:$(TAG)
