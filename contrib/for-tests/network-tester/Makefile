all: push

# By default, pushes to a githash.  To override:
# make push TAG=<version>
# Next tag is 1.5

TAG = $(shell git log -1 --format="format:%H")

webserver: webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./webserver.go

container: webserver
	docker build -t gcr.io/google_containers/nettest:$(TAG) .

push: container
	gcloud preview docker push gcr.io/google_containers/nettest:$(TAG)

clean:
	rm -f webserver
