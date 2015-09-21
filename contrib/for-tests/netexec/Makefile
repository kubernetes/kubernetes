.PHONY: all netexec image push clean

TAG = 1.1
PREFIX = gcr.io/google_containers

all: push

netexec: netexec.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./netexec.go

image: netexec
	docker build -t $(PREFIX)/netexec:$(TAG) .

push: image
	gcloud docker push $(PREFIX)/netexec:$(TAG)

clean:
	rm -f netexec
