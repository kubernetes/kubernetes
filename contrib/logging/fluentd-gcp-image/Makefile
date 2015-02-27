# The build rule builds a Docker image that logs all Docker contains logs to
# Google Compute Platform using the Cloud Logging API. The push rule pushes
# the image to DockerHub.
# Satnam Singh (satnam@google.com)

.PHONY:	build push


TAG = 1.0

build:	
	sudo docker build -t kubernetes/fluentd-gcp:$(TAG) .

push:	
	sudo docker push kubernetes/fluentd-gcp:$(TAG)

