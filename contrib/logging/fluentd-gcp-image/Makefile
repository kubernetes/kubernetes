# The build rule builds a Docker image that logs all Docker contains logs to
# Google Compute Platform using the Cloud Logging API. The push rule pushes
# the image to DockerHub.
# Satnam Singh (satnam@google.com)

.PHONY:	build push

build:	
	sudo docker build -t kubernetes/fluentd-gcp .

push:	
	sudo docker push kubernetes/fluentd-gcp

