all: push

TAG = 0.4

container:
	docker build -t gcr.io/google_containers/volume-nfs . # Build new image and automatically tag it as latest
	docker tag gcr.io/google_containers/volume-nfs gcr.io/google_containers/volume-nfs:$(TAG) # Add the version tag to the latest image 

push: container
	gcloud docker push gcr.io/google_containers/volume-nfs # Push image tagged as latest to repository
	gcloud docker push gcr.io/google_containers/volume-nfs:$(TAG) # Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)

clean:
