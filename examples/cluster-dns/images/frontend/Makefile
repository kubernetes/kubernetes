TAG = v1
PREFIX = gcr.io/google_containers
IMAGE = example-dns-frontend

all: push

image:
	docker build -t $(PREFIX)/$(IMAGE):$(TAG) .

push: image
	gcloud docker push $(PREFIX)/$(IMAGE)

clean:
