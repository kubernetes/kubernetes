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

.PHONY:	elasticsearch_logging_discovery build push

# The current value of the tag to be used for building and
# pushing an image to gcr.io
TAG = 1.9

build:	elasticsearch_logging_discovery
	docker build -t gcr.io/google_containers/elasticsearch:$(TAG) .

push:
	gcloud docker push gcr.io/google_containers/elasticsearch:$(TAG)

elasticsearch_logging_discovery:
	go build elasticsearch_logging_discovery.go

clean:
	rm elasticsearch_logging_discovery
