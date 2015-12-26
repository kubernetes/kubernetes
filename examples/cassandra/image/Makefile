# build the cassandra image.

VERSION=v7

all: build

kubernetes-cassandra.jar: ../java/* ../java/src/io/k8s/cassandra/*.java
	cd ../java && mvn package
	mv ../java/target/kubernetes-cassandra*.jar kubernetes-cassandra.jar
	cd ../java && mvn clean

build: kubernetes-cassandra.jar
	docker build -t gcr.io/google_containers/cassandra:${VERSION} .

push: build
	gcloud docker push gcr.io/google_containers/cassandra:${VERSION}

.PHONY: all build push
