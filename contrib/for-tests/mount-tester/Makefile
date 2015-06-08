all: push

mt: mt.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./mt.go

image: mt
	sudo docker build -t kubernetes/mounttest .

push: image
	sudo docker push kubernetes/mounttest

clean:
	rm -f mt
