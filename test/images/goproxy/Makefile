all: push

TAG = 0.1

goproxy: goproxy.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./goproxy.go

image: goproxy
	docker build -t gcr.io/google_containers/goproxy:$(TAG) .

push: image
	gcloud docker push gcr.io/google_containers/goproxy:$(TAG)

clean:
	rm -f goproxy
