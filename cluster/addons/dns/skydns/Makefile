all: skydns

skydns:
	CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' github.com/skynetservices/skydns

container: skydns
	sudo docker build -t gcr.io/google_containers/skydns .

push:
	sudo gcloud docker push gcr.io/google_containers/skydns

clean:
	rm -f skydns
