# build the cassandra image.

VERSION=v7

all: build

# TODO: build the jar and copy it in
build:
        docker build -t gcr.io/google_containers/cassandra:${VERSION} .

push: build
        gcloud docker push gcr.io/google_containers/cassandra:${VERSION}

.PHONY: all
