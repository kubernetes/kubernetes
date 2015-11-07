TAG = 0.2
PREFIX = gcr.io/google_containers

all: push

container: image

image:
	docker build -t $(PREFIX)/volume-gluster . # Build new image and automatically tag it as latest
	docker tag $(PREFIX)/volume-gluster $(PREFIX)/volume-gluster:$(TAG)  # Add the version tag to the latest image 

push: image
	gcloud docker push $(PREFIX)/volume-gluster # Push image tagged as latest to repository
	gcloud docker push $(PREFIX)/volume-gluster:$(TAG) # Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)

clean:
