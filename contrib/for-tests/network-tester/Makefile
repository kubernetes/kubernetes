all: push

webserver: webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./webserver.go

container: webserver
	sudo docker build -t kubernetes/nettest .

push: container
	sudo docker push kubernetes/nettest

clean:
	rm -f webserver
