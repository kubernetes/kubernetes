all: push

TAG = 1.0

portforwardtester: portforwardtester.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./portforwardtester.go

image: portforwardtester
	docker build -t gcr.io/google_containers/portforwardtester:$(TAG) .

push: image
	gcloud docker push gcr.io/google_containers/portforwardtester:$(TAG)

clean:
	rm -f portforwardtester
