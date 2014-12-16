all: push

server: server.go
	CGO_ENABLED=0 GOOS=linux go build -a -tags netgo -ldflags '-w' ./server.go

container: server
	docker build -t kubernetes/liveness .

push: container
	docker push kubernetes/liveness

clean:
	rm -f server
