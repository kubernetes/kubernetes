# build Kubemark image from currently built binaries containing both 'real' master and Hollow Node.
# This makefile assumes that the kubemark binary is present in this directory.

all:
	docker build -t gcr.io/$(PROJECT)/kubemark .
	gcloud docker push gcr.io/$(PROJECT)/kubemark

.PHONY: all
