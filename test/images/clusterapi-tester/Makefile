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

all: push

# 0.0 shouldn't clobber any released builds
TAG = 1.0
PREFIX = gcr.io/google_containers/clusterapi-tester

main: main.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' -o main ./main.go

container: main
	docker build -t $(PREFIX):$(TAG) .

push: container
	gcloud docker push $(PREFIX):$(TAG)

clean:
	rm -f main
