all: skydns

skydns:
	CGO_ENABLED=0 go build -a --ldflags '-w' github.com/skynetservices/skydns

container: skydns
	sudo docker build -t kubernetes/skydns .

push:
	sudo docker push kubernetes/skydns

clean:
	rm -f skydns
