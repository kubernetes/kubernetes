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

# build the kubelet image.

# TODO: figure out the best base image
# TODO: figure out how to best source the nsenter binary, if necessary
# TODO: figure out how to make a single source of version info for these
#       images
VERSION=v0.16
BIN_DIR=bin/linux/amd64
LOCAL_OUTPUT_DIR=../../../_output/local

release: clean
	curl -O https://storage.googleapis.com/kubernetes-release/release/${VERSION}/${BIN_DIR}/kubelet
	docker build -t gcr.io/google_containers/kubelet:${VERSION} .
	gcloud docker push gcr.io/google_containers/kubelet:${VERSION}

local: clean
	cp ${LOCAL_OUTPUT_DIR}/${BIN_DIR}/kubelet .
	docker build -t gcr.io/google_containers/kubelet .

clean:
	rm -f kubelet

.PHONY: release
