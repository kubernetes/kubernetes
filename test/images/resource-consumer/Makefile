TAG = beta
PREFIX = gcr.io/google_containers

all: clean consumer

consumer:
	CGO_ENABLED=0 godep go build -a -installsuffix cgo --ldflags '-w' -o consume-cpu/consume-cpu ./consume-cpu/consume_cpu.go
	CGO_ENABLED=0 godep go build -a -installsuffix cgo --ldflags '-w' -o consumer .

container: image

image:
	sudo docker build -t $(PREFIX)/resource_consumer:$(TAG) .

run_container:
	docker run --publish=8080:8080 $(PREFIX)/resource_consumer:$(TAG)

push:
	@echo "This image is not meant to be pushed."

clean:
	rm -f consumer
	rm -f consume-cpu/consume-cpu
