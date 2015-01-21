all: push

test-webserver: test-webserver.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./test-webserver.go

container: test-webserver
	sudo docker build -t kubernetes/test-webserver .

push: container
	sudo docker push kubernetes/test-webserver

clean:
	rm -f test-webserver
