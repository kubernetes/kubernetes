all: push

TAG = 0.1

ep: ep.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./ep.go

image: ep
	sudo docker build -t kubernetes/eptest:$(TAG) .

push: image
	sudo docker push kubernetes/eptest:$(TAG)

clean:
	rm -f ep
