TAG = 0.3
PREFIX = gcr.io/google_containers

all: push

image:
	sudo docker build -t $(PREFIX)/mounttest-user:$(TAG) .

push: image
	gcloud docker push $(PREFIX)/mounttest-user:$(TAG)

clean:
