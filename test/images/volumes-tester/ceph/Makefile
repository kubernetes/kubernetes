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
	docker build -t $(PREFIX)/volume-ceph . # Build new image and automatically tag it as latest
	docker tag $(PREFIX)/volume-ceph $(PREFIX)/volume-ceph:$(TAG)  # Add the version tag to the latest image

push: image
	gcloud docker push $(PREFIX)/volume-ceph # Push image tagged as latest to repository
	gcloud docker push $(PREFIX)/volume-ceph:$(TAG) # Push version tagged image to repository (since this image is already pushed it will simply create or update version tag)

clean:
