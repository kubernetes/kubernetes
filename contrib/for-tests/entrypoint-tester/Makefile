all: push

# By default, pushes to a githash.  To override:
# make push TAG=<version>
# Next tag is 0.2

TAG = $(shell git log -1 --format="format:%H")

ep: ep.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./ep.go

image: ep
	sudo docker build -t kubernetes/eptest:$(TAG) .

push: image
	sudo docker push kubernetes/eptest:$(TAG)

clean:
	rm -f ep
