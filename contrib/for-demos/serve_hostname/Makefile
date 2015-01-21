all: serve_hostname

TAG = 1.0

serve_hostname: serve_hostname.go
	CGO_ENABLED=0 go build -a --ldflags '-w' ./serve_hostname.go

container: serve_hostname
	sudo docker build -t kubernetes/serve_hostname:$(TAG) .

push:
	sudo docker push kubernetes/serve_hostname:$(TAG)

clean:
	rm -f serve_hostname
