# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
	gcloud docker -- push $(PREFIX)/volume-iscsi
	# Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)
	gcloud docker -- push $(PREFIX)/volume-iscsi:$(TAG)

clean:
