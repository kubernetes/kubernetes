all: push

TAG = 1.1

webserver: webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./webserver.go

container: webserver
	docker build -t gcr.io/google_containers/nettest:$(TAG) .

push: container
	gcloud preview docker push gcr.io/google_containers/nettest:$(TAG)

clean:
	rm -f webserver
