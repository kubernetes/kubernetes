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

IMG = gcr.io/google-containers/kubekins-e2e
TAG = $(shell date +v%Y%m%d)

e2e:
	cp ../../e2e.go .

e2e-runner:
	cp ../e2e-runner.sh .

sh2ju:
	cp ../../../third_party/forked/shell2junit/sh2ju.sh .

upload-to-gcs.sh:
	cp ../upload-to-gcs.sh .

clean:
	rm -rf e2e.go e2e-runner.sh sh2ju.sh upload-to-gcs.sh

all: push

build: e2e e2e-runner sh2ju upload-to-gcs
	docker build -t $(IMG):$(TAG) .
	docker tag -f $(IMG):$(TAG) $(IMG):latest
	@echo Built $(IMG):$(TAG) and tagged with latest

push: build
	gcloud docker push $(IMG):$(TAG)
	gcloud docker push $(IMG):latest
	@echo Pushed $(IMG) with :latest and :$(TAG) tags
