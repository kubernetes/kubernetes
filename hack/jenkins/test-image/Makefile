all: push

TAG = 0.1

container:
	docker build -t gcr.io/google_containers/kubekins-test .
	docker tag gcr.io/google_containers/kubekins-test gcr.io/google_containers/kubekins-test:$(TAG)

push: container
	gcloud docker push gcr.io/google_containers/kubekins-test # Push image tagged as latest to repository
	gcloud docker push gcr.io/google_containers/kubekins-test:$(TAG) # Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)

clean:
