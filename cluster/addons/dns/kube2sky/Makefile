all: kube2sky

kube2sky: kube2sky.go
	CGO_ENABLED=0 godep go build -a -installsuffix cgo --ldflags '-w' ./kube2sky.go

container: kube2sky
	docker build -t gcr.io/google_containers/kube2sky .

push:
	gcloud preview docker push gcr.io/google_containers/kube2sky

clean:
	rm -f kube2sky
