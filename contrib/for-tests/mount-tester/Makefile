all: push

# By default, pushes to a githash.  To override:
# make push TAG=<version>
# Next TAG is 0.2

TAG = $(shell git log -1 --format="format:%H")

mt: mt.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./mt.go

image: mt
	sudo docker build -t kubernetes/mounttest:$(TAG) .

push: image
	sudo docker push kubernetes/mounttest:$(TAG)

clean:
	rm -f mt
