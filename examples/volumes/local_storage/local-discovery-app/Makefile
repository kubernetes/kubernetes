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
# `make local-discovery` will build local-discovery.
# `make container` will build a container-- you must supply a tag.
# `make push` will push the container-- you must supply a tag.

GOARCH?=$(shell go env GOARCH)
GOOS?=$(shell go env GOOS)
TAG="v1.3"

local-discovery:
	CGO_ENABLED=0 go build -a -ldflags '-extldflags "-static"' local-discovery.go

container:
	docker build --pull -t gcr.io/msau-gke-dev/local-discovery:$(TAG) .

push: container
	gcloud docker -- push gcr.io/msau-gke-dev/local-discovery:$(TAG)

clean:
	rm -f local-discovery
