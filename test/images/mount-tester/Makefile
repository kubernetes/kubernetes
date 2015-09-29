all: push

TAG = 0.4

mt: mt.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./mt.go

image: mt
	sudo docker build -t gcr.io/google_containers/mounttest:$(TAG) .

push: image
	gcloud docker push gcr.io/google_containers/mounttest:$(TAG)

clean:
	rm -f mt
