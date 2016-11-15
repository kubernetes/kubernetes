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
# `make porter` will build porter.
# `make tag` will suggest a tag.
# `make image` will build an image-- you must supply a tag.
# `make push` will push the image-- you must supply a tag.

# This image does not tag in the normal way
# TAG =
PREFIX = gcr.io/google_containers
SUGGESTED_TAG = $(shell git rev-parse --verify HEAD)

porter: porter.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./porter.go

tag:
	@echo "If all relevant changes are committed, suggest using TAG=$(SUGGESTED_TAG)"
	@echo "$$ make container TAG=$(SUGGESTED_TAG)"
	@echo "or"
	@echo "$$ make push TAG=$(SUGGESTED_TAG)"

container: image

image:
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' after committing changes to see a suggestion))
	docker build -t $(PREFIX)/porter:$(TAG) .

push:
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' after committing changes to see a suggestion))
	gcloud docker -- push $(PREFIX)/porter:$(TAG)

clean:
	rm -f porter
