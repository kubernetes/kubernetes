all: push

TAG = 0.1

mt: mt.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./mt.go

image: mt
	sudo docker build -t kubernetes/mounttest:$(TAG) .

push: image
	sudo docker push kubernetes/mounttest:$(TAG)

clean:
	rm -f mt
