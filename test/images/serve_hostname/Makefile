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

all: serve_hostname

TAG = v1.4
PREFIX = gcr.io/google_containers
TEST_PREFIX = b.gcr.io/k8s_authenticated_test

serve_hostname: serve_hostname.go
	CGO_ENABLED=0 go build -a -installsuffix cgo --ldflags '-w' ./serve_hostname.go

container: serve_hostname
	docker build -t $(PREFIX)/serve_hostname:$(TAG) .
	if [ -n "$(TEST_PREFIX)" ]; then \
	    docker tag -f $(PREFIX)/serve_hostname:$(TAG) $(TEST_PREFIX)/serve_hostname:$(TAG); \
	fi

push:
	gcloud docker push $(PREFIX)/serve_hostname:$(TAG)
	if [ -n "$(TEST_PREFIX)" ]; then \
	    gcloud docker push $(TEST_PREFIX)/serve_hostname:$(TAG); \
	fi

clean:
	rm -f serve_hostname
