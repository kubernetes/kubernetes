all: serve_hostname

TAG = 1.1

serve_hostname: serve_hostname.go
	CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' ./serve_hostname.go

container: serve_hostname
	sudo docker build -t kubernetes/serve_hostname:$(TAG) .

push:
	sudo docker push kubernetes/serve_hostname:$(TAG)

clean:
	rm -f serve_hostname
