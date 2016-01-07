TAG = 1.6
PREFIX = gcr.io/google_containers

all: push

webserver: webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./webserver.go

container: image

image: webserver
	docker build -t $(PREFIX)/nettest:$(TAG) .

push: image
	gcloud docker push $(PREFIX)/nettest:$(TAG)

clean:
	rm -f webserver
