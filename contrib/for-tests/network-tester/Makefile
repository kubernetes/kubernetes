all: push

TAG = 1.1

webserver: webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./webserver.go

container: webserver
	sudo docker build -t lmktfy/nettest:$(TAG) .

push: container
	sudo docker push lmktfy/nettest:$(TAG)

clean:
	rm -f webserver
