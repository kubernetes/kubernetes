all: serve_hostname

serve_hostname: serve_hostname.go
	CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' ./serve_hostname.go

container: serve_hostname
	sudo docker build -t kubernetes/serve_hostname .

push:
	sudo docker push kubernetes/serve_hostname

clean:
	rm -f serve_hostname
