# By default, pushes to a githash.  To override:
# make push TAG=<version>
# Next TAG=1.0

TAG = $(shell git log -1 --format="format:%H")

all: skydns

skydns:
	CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' github.com/skynetservices/skydns

container: skydns
	sudo docker build -t gcr.io/google_containers/skydns:$(TAG) .

push:
	sudo gcloud preview docker push gcr.io/google_containers/skydns:$(TAG)

clean:
	rm -f skydns
