# This image does not tag
#TAG =
PREFIX = gcr.io/google_containers

all: push

image:
	docker build -t $(PREFIX)/dnsutils .

push: image
	gcloud docker push $(PREFIX)/dnsutils

clean:
