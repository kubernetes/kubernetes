# The build rule builds a Docker image that logs all Docker contains logs to
# Google Compute Platform using the Cloud Logging API. The push rule pushes
# the image to DockerHub.

# Procedure for change:
# 1. Bump the tag number.
# 2. Create an image for the kubernetes user on Dockerhub and push there: make kbuild kpush
# 3. Temporarily adjust any config files etc. that reference gcr.io/google_containers/fluentd-gcp:$(TAG)
#    to use kubernetes/fluentd-gcp:$(TAG) instead.
# 4. Test etc.
# 5. Issue PR.
# 6. When PR is approved make the gcr.io version of the image: make build push
# 7. Revert the referendes to kubernetes/fluentd-gcp:$(TAG) to gcr.io/google_containers/fluentd-gcp:$(TAG) 

.PHONY:	kbuild kpush

TAG = 1.14

# Rules for building the test image for deployment to Dockerhub with user kubernetes.

kbuild:
	docker build -t kubernetes/fluentd-gcp:$(TAG) .


kpush:
	docker push kubernetes/fluentd-gcp:$(TAG)


# Rules for building the real image for deployment to gcr.io

build:
	docker build -t gcr.io/google_containers/fluentd-gcp:$(TAG) .


push:
	gcloud docker push gcr.io/google_containers/fluentd-gcp:$(TAG)
