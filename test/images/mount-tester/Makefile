TAG = 0.5
PREFIX = gcr.io/google_containers

all: push

mt: mt.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./mt.go

image: mt
	sudo docker build -t $(PREFIX)/mounttest:$(TAG) .

push: image
	gcloud docker push $(PREFIX)/mounttest:$(TAG)

clean:
	rm -f mt
