all: push

TAG = 0.1

container:
	# Build new image and automatically tag it as latest
	docker build -t gcr.io/google_containers/volume-rbd .
	# Add the version tag to the latest image 
	docker tag gcr.io/google_containers/volume-rbd gcr.io/google_containers/volume-rbd:$(TAG)

block:
	# Create block.tar.gz with ext2 block device with index.html inside.
	# block.tar.gz is already available in git and users don't need to
	# regenerate it, this target is here just for reference.
	# Run as root!
	./create_block.sh

push: container
	# Push image tagged as latest to repository
	gcloud docker push gcr.io/google_containers/volume-rbd
	# Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)
	gcloud docker push gcr.io/google_containers/volume-rbd:$(TAG)
