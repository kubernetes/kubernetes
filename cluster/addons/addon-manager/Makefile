# Copyright 2016 The Kubernetes Authors All rights reserved.
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

IMAGE=gcr.io/google-containers/kube-addon-manager
VERSION=v1
KUBECTL_VERSION=v1.2.3

.PHONY: build push container

build: kubectl
	docker build -t "$(IMAGE):$(VERSION)" .

kubectl:
	curl "https://storage.googleapis.com/kubernetes-release/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
		-o kubectl
	chmod +x kubectl

push: build
	gcloud docker push "$(IMAGE):$(VERSION)"

clean:
	rm kubectl
	docker rmi -f "$(IMAGE):$(VERSION)"
