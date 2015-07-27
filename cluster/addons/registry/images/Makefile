.PHONY: build push vet test clean

TAG = 0.2.3
REPO = uluyol/kube-registry-proxy

build:
	docker build -t $(REPO):$(TAG) .

push:
	docker push $(REPO):$(TAG)
