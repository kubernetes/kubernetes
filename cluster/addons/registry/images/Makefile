.PHONY: build push vet test clean

TAG = 0.3
REPO = gcr.io/google_containers/kube-registry-proxy

build:
	docker build -t $(REPO):$(TAG) .

push:
	gcloud docker push $(REPO):$(TAG)
