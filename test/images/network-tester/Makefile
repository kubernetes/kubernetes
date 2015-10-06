all: push

# Set this to the *next* version to prevent accidentally overwriting the existing image.
TAG = 1.6

webserver: webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./webserver.go

container: webserver
	docker build -t beta.gcr.io/google_containers/nettest:$(TAG) .

push: container
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/nettest:$(TAG)

clean:
	rm -f webserver
