all: resource_consumer

resource_consumer: 
	CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' .

clean:
	rm -f resource-consumer
