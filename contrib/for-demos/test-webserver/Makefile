all: push

# By default, pushes to a githash.  To override:
# make push TAG=<version>

TAG = $(shell git log -1 --format="format:%H")


test-webserver: test-webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./test-webserver.go

container: test-webserver
	sudo docker build -t gcr.io/google_containers/test-webserver:$(TAG) .

push: container
	gcloud preview docker push gcr.io/google_containers/test-webserver:$(TAG)

clean:
	rm -f test-webserver
