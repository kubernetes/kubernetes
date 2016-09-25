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

# Use:
#
# `make kubectl` will build kubectl.
# `make tag` will suggest a tag.
# `make container` will build a container-- you must supply a tag.
# `make push` will push the container-- you must supply a tag.

GOARCH?=$(shell go env GOARCH)
GOOS?=$(shell go env GOOS)

kubectl:
	make -C ../../ WHAT=cmd/kubectl KUBE_STATIC_OVERRIDES="kubectl"; \
	cp ../../_output/local/bin/$(GOOS)/$(GOARCH)/kubectl .

.tag: kubectl
	./kubectl version --client | grep -o 'GitVersion:"[^"]*"' | sed 's/[^"]*"\([^"+]*\).*/\1/' > .tag

tag: .tag
	@echo "Suggest using TAG=$(shell cat .tag)"
	@echo "$$ make container TAG=$(shell cat .tag)"
	@echo "or"
	@echo "$$ make push TAG=$(shell cat .tag)"

container:
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' to see a suggestion))
	docker build -t gcr.io/google_containers/kubectl:$(TAG) .

push: container
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' to see a suggestion))
	gcloud docker push gcr.io/google_containers/kubectl:$(TAG)

clean:
	rm -f kubectl
	rm -f .tag
