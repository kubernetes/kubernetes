all: push

TAG = 0.2

image:
	sudo docker build -t gcr.io/google_containers/mounttest-user:$(TAG) .

push: image
	gcloud docker push gcr.io/google_containers/mounttest-user:$(TAG)
