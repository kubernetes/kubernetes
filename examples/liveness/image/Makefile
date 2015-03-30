all: push

server: server.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./server.go

container: server
	docker build -t lmktfy/liveness .

push: container
	docker push lmktfy/liveness

clean:
	rm -f server
