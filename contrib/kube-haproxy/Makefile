all: push

# Set this to the *next* version to prevent accidentally overwriting the existing image.
# Next tag=0.3
# Usage:
#  tag with the current git hash:
#   make TAG=`git log -1 --format="%H"`
#  tag with a formal version
#   make TAG=0.2

kube-haproxy: kube-haproxy.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./kube-haproxy.go

container: kube-haproxy
	docker build -t gcr.io/google_containers/kube-haproxy:$(TAG) .

push: container
	gcloud docker push gcr.io/google_containers/kube-haproxy:$(TAG)

clean:
	rm -f kube-haproxy
