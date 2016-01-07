TAG = 0.1
PREFIX = gcr.io/google_containers

all: push

container: image

image:
	# Build new image and automatically tag it as latest
	docker build -t $(PREFIX)/volume-iscsi .
	# Add the version tag to the latest image
	docker tag $(PREFIX)/volume-iscsi $(PREFIX)/volume-iscsi:$(TAG)

block:
	# Create block.tar.gz with ext2 block device with index.html inside.
	# block.tar.gz is already available in git and users don't need to
	# regenerate it, this target is here just for reference.
	# Run as root!
	./create_block.sh

push: image
	# Push image tagged as latest to repository
	gcloud docker push $(PREFIX)/volume-iscsi
	# Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)
	gcloud docker push $(PREFIX)/volume-iscsi:$(TAG)

clean:
