all: push

# Set this to the *next* version to prevent accidentally overwriting the existing image.
# Next tag=1.2
# Usage:
#  tag with the current git hash:
#   make TAG=`git log -1 --format="%H"`
#  tag with a formal version
#   make TAG=1.2

podmaster: podmaster.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./podmaster.go

container: podmaster
	docker build -t gcr.io/google_containers/podmaster:$(TAG) .

push: container
	gcloud preview docker push gcr.io/google_containers/podmaster:$(TAG)

clean:
	rm -f podmaster
