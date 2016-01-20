TAG = 0.1
PREFIX = gcr.io/google_containers

all: push

goproxy: goproxy.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./goproxy.go

image: goproxy
	docker build -t $(PREFIX)/goproxy:$(TAG) .

push: image
	gcloud docker push $(PREFIX)/goproxy:$(TAG)

clean:
	rm -f goproxy
