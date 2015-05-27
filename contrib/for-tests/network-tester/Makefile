all: push

# Set this to the *next* version to prevent accidentally overwriting the existing image.
TAG = 1.4

webserver: webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./webserver.go

container: webserver
	docker build -t gcr.io/google_containers/nettest:$(TAG) .

push: container
	gcloud preview docker push gcr.io/google_containers/nettest:$(TAG)

clean:
	rm -f webserver
