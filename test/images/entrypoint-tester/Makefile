TAG = 0.1
PREFIX = kubernetes

all: push

ep: ep.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./ep.go

image: ep
	sudo docker build -t $(PREFIX)/eptest:$(TAG) .

push: image
	sudo docker push $(PREFIX)/eptest:$(TAG)

clean:
	rm -f ep
