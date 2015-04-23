.PHONY:	build push

TAG = 1.0

build:
	docker build -t gcr.io/google_containers/fluentd-sidecar-es:$(TAG) .

push:
	gcloud preview docker push gcr.io/google_containers/fluentd-sidecar-es:$(TAG)
