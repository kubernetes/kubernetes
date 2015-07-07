.PHONY:	build push

TAG = 1.1

build:
	docker build -t gcr.io/google_containers/fluentd-sidecar-es:$(TAG) .

push:
	gcloud docker push gcr.io/google_containers/fluentd-sidecar-es:$(TAG)
