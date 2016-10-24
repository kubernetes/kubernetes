all: push

# 0.0 shouldn't clobber any released builds
TAG ?= 0.1
# TODO: Make cross platform, push to gcr.io/google_containers
PREFIX ?= bprashanth/kube-dnsmasq-amd64
GCLOUD ?= gcloud
SRC = dnsmasq-babysitter.go

babysitter: $(SRC)
	CGO_ENABLED=0 GOOS=linux godep go build -a -installsuffix cgo -ldflags '-w' -o $@ $(SRC)

container: babysitter
	docker build -t $(PREFIX):$(TAG) .

push: container
	$(GCLOUD) docker push $(PREFIX):$(TAG)

clean:
	docker rmi -f $(PREFIX):$(TAG) || true
	rm -f babysitter

.PHONY: container push
