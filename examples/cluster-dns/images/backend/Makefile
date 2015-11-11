TAG = v1
PREFIX = gcr.io/google_containers
IMAGE = example-dns-backend

all: push

image:
	docker build -t $(PREFIX)/$(IMAGE):$(TAG) .

push: image
	gcloud docker push $(PREFIX)/$(IMAGE)

clean:
