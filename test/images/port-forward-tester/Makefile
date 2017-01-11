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

TAG = 1.1
PREFIX = gcr.io/google_containers

all: push

portforwardtester: portforwardtester.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./portforwardtester.go

image: portforwardtester
	docker build --pull -t $(PREFIX)/portforwardtester:$(TAG) .

push: image
	gcloud docker -- push $(PREFIX)/portforwardtester:$(TAG)

clean:
	rm -f portforwardtester
