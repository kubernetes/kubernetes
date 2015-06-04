all: push

# By default, pushes to a githash.  To override:
# make push TAG=<version>
# next TAG=1.2

TAG = $(shell git log -1 --format="format:%H")

podmaster: podmaster.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./podmaster.go

container: podmaster
	docker build -t gcr.io/google_containers/podmaster:$(TAG) .

push: container
	gcloud preview docker push gcr.io/google_containers/podmaster:$(TAG)

clean:
	rm -f podmaster
