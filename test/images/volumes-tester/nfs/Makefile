all: push

TAG = 0.3

container:
	docker build -t beta.gcr.io/google_containers/volume-nfs . # Build new image and automatically tag it as latest
	docker tag beta.gcr.io/google_containers/volume-nfs beta.gcr.io/google_containers/volume-nfs:$(TAG) # Add the version tag to the latest image 

push: container
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/volume-nfs # Push image tagged as latest to repository
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/volume-nfs:$(TAG) # Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)

clean:
