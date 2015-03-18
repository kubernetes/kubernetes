all: push

TAG = 1.1

webserver: webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./webserver.go

container: webserver
	sudo docker build -t kubernetes/nettest:$(TAG) .

push: container
	sudo docker push kubernetes/nettest:$(TAG)

clean:
	rm -f webserver
