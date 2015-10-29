TAG = 1.0
PREFIX = gcr.io/google_containers

all: push

portforwardtester: portforwardtester.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./portforwardtester.go

image: portforwardtester
	docker build -t $(PREFIX)/portforwardtester:$(TAG) .

push: image
	gcloud docker push $(PREFIX)/portforwardtester:$(TAG)

clean:
	rm -f portforwardtester
