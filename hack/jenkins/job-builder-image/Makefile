TAG = 1

all:
	docker build -t gcr.io/google_containers/kubekins-job-builder:$(TAG) .
	gcloud docker push gcr.io/google_containers/kubekins-job-builder:$(TAG)
