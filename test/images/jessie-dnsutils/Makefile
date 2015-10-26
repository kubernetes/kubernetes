# This image does not tag
#TAG =
PREFIX = gcr.io/google_containers

all: push

image:
	docker build -t $(PREFIX)/jessie-dnsutils .

push: image
	gcloud docker push $(PREFIX)/jessie-dnsutils

clean:
