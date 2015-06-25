all: push

test-webserver: test-webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./test-webserver.go

container: test-webserver
	sudo docker build -t gcr.io/google_containers/test-webserver .

push: container
	gcloud docker push gcr.io/google_containers/test-webserver

clean:
	rm -f test-webserver
