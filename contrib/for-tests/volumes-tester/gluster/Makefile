all: push

TAG = 0.1

container:
	docker build -t gcr.io/google_containers/volume-gluster:$(TAG) .

push: container
	gcloud preview docker push gcr.io/google_containers/volume-gluster:$(TAG)

clean:
