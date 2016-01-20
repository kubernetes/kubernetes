.PHONY: all image push clean

TAG = 1.2
PREFIX = gcr.io/google_containers


all: push

image:
	docker build -t $(PREFIX)/hostexec:$(TAG) .

push: image
	gcloud docker push $(PREFIX)/hostexec:$(TAG)

clean:
	rm -f hostexec
