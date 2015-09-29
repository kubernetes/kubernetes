all: push

TAG = 0.2

container:
	docker build -t gcr.io/google_containers/volume-gluster . # Build new image and automatically tag it as latest
	docker tag gcr.io/google_containers/volume-gluster gcr.io/google_containers/volume-gluster:$(TAG)  # Add the version tag to the latest image 

push: container
	gcloud docker push gcr.io/google_containers/volume-gluster # Push image tagged as latest to repository
	gcloud docker push gcr.io/google_containers/volume-gluster:$(TAG) # Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)

clean:
