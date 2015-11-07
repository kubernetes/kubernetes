TAG = 0.1
PREFIX = gcr.io/google_containers

all: push

container: image

image:
	docker build -t $(PREFIX)/volume-ceph . # Build new image and automatically tag it as latest
	docker tag $(PREFIX)/volume-ceph $(PREFIX)/volume-ceph:$(TAG)  # Add the version tag to the latest image

push: image
	gcloud docker push $(PREFIX)/volume-ceph # Push image tagged as latest to repository
	gcloud docker push $(PREFIX)/volume-ceph:$(TAG) # Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)

clean:
