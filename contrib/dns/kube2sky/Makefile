all: kube2sky

kube2sky: kube2sky.go
	CGO_ENABLED=0 go build -a --ldflags '-w' ./kube2sky.go

container: kube2sky
	sudo docker build -t kubernetes/kube2sky .

push:
	sudo docker push kubernetes/kube2sky

clean:
	rm -f kube2sky
