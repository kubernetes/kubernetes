# Copyright 2017 The Kubernetes Authors.
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

.PHONY:	build push

# TAG is the version to build and push to.
PREFIX = gcr.io/qlee_dev
NAME = dind-test-base
ARCH = amd64
TAG = 0.1

IMAGE_NO_TAG = ${PREFIX}/${NAME}-${ARCH}
IMAGE = ${IMAGE_NO_TAG}:$(TAG)

build:
	# We want to keep these files with their licenses, but Docker builds
	# need everything under a root dir. Docker can't follow symlinks, and
	# the whole k8s repo is too large. Instead we move the files in for the
	# build, and then clean up after.
	cp ../../../third_party/moby/dind .
	cp ../../../third_party/moby/dockerd-entrypoint.sh .
	# We explicitly add "--pull" flag to always fetch the latest version
	# of the base image. This is necessary to avoid using cached local
	# versions of image e.g. when updating insecure base images.
	docker build --pull -t ${IMAGE} .
	rm -f dind
	rm -f dockerd-entrypoint.sh

push: build
	gcloud docker -- push ${IMAGE}

clean:
	rm -f dind
	rm -f dockerd-entrypoint.sh
