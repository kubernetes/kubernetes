all: push

server: server.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./server.go

container: server
	docker build -t beta.gcr.io/google_containers/liveness .

push: container
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/liveness

clean:
	rm -f server
