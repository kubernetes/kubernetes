all: push

TAG = 0.2

image:
	sudo docker build -t beta.gcr.io/google_containers/mounttest-user:$(TAG) .

push: image
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/mounttest-user:$(TAG)
