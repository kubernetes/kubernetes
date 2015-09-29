# build Kubemark image from currently built binaries containing both 'real' master and Hollow Node

all:
	cp ../../../_output/release-tars/kubernetes-server-linux-amd64.tar.gz .
	docker build -t gcr.io/$(PROJECT)/kubemark .
	gcloud docker push gcr.io/$(PROJECT)/kubemark
	rm kubernetes-server-linux-amd64.tar.gz

.PHONY: all
