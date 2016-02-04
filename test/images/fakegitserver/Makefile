.PHONY:	all image push clean

TAG = 0.1
PREFIX = gcr.io/google_containers
IMAGE = fakegitserver
ARCH = amd64

image:
	./prepare.sh $(ARCH)
	docker build -t $(PREFIX)/$(IMAGE):$(TAG) .

push: image
	gcloud docker push $(PREFIX)/$(IMAGE):$(TAG)

all: push

clean:
	rm -f gitserver GITHASH.txt
