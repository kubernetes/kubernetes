all: clean consumer

TAG = alpha

consumer:
	CGO_ENABLED=0 godep go build -a -installsuffix cgo --ldflags '-w' -o consume-cpu/consume-cpu ./consume-cpu/consume_cpu.go
	CGO_ENABLED=0 godep go build -a -installsuffix cgo --ldflags '-w' -o consumer .

container:
	sudo docker build -t gcr.io/google_containers/resource_consumer:$(TAG) .	

run_container:
	docker run --publish=8080:8080 gcr.io/google_containers/resource_consumer:$(TAG)

clean:
	rm -f consumer
	rm -f consume-cpu/consume-cpu
