TAG = 0.4
PREFIX = gcr.io/google_containers

all: push

container: image

image:
	docker build -t $(PREFIX)/volume-nfs . # Build new image and automatically tag it as latest
	docker tag $(PREFIX)/volume-nfs $(PREFIX)/volume-nfs:$(TAG) # Add the version tag to the latest image

push: image
	gcloud docker push $(PREFIX)/volume-nfs # Push image tagged as latest to repository
	gcloud docker push $(PREFIX)/volume-nfs:$(TAG) # Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)

clean:
