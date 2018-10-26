# Copyright 2017 The Kubernetes Authors.
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

ARCH ?= amd64
REGISTRY ?= gcr.io
PROJECT ?= $(shell whoami)-k8s-dev
VERSION ?= $(shell cat build/version.txt)
IMAGE ?= $(shell docker image ls | egrep "${PROJECT}/tomcat" | sort +2 -3 | tail -1 | sed -E 's/\s+/ /g' | cut -d" " -f1,2 | sed -E 's/\s/:/')

bin/eventLoad: eventLoad.go
	go build -o bin/eventLoad eventLoad.go

bin/eventCheck: eventCheck.go
	go build -o bin/eventCheck eventCheck.go

build/version.txt: setup
	@date +%s > build/version.txt

tmp/sample.war: setup
	wget -O tmp/sample.war https://tomcat.apache.org/tomcat-9.0-doc/appdev/sample/sample.war

setup:
	mkdir -p bin
	mkdir -p build
	mkdir -p tmp

list_setup:
	@echo kubectl create -f artifacts/templates/namespace.yaml
	@echo kubectl create -f artifacts/templates/secret.yaml
	@echo kubectl create -f artifacts/templates/networkPolicy.yaml
	@echo kubectl create -f artifacts/templates/pv.yaml
	@echo kubectl create -f artifacts/templates/pvc.yaml
	@echo kubectl create -f artifacts/templates/deployment.yaml
	@echo kubectl create -f artifacts/templates/services.yaml
	@echo kubectl create -f artifacts/templates/ingress.yaml

build_test: setup bin/eventLoad bin/eventCheck

push_test: bin/eventLoad bin/eventCheck
	gcloud compute scp --project "${PROJECT}"  --zone "${ZONE}" ./bin/eventLoad kubernetes-master:/tmp
	gcloud compute scp --project "${PROJECT}"  --zone "${ZONE}" ./bin/eventCheck kubernetes-master:/tmp
	@echo /tmp/eventLoad --kubeconfig "" --image ${IMAGE} --concurrentSites 2 --numSites 6 --inCluster=true --killList="kube-apiserver,kube-scheduler,kube-controller-manager" --killInterval=180

list_test:
	@echo Instructions for testing project ${PROJECT}
	@echo ./bin/eventLoad --kubeconfig ~/.kube/config --image ${IMAGE}

generate_images: build/version.txt tmp/sample.war
	docker build -q -t ${REGISTRY}/${PROJECT}/tomcat-${ARCH}:${VERSION} -f artifacts/images/tomcat-${ARCH}-dockerfile . > /dev/null
	docker images -q ${REGISTRY}/${PROJECT}/tomcat-${ARCH}:${VERSION} >> build/${REGISTRY}_${PROJECT}_tomcat-${ARCH}_${VERSION}-container

publish_images: generate_images
	gcloud docker -- push ${REGISTRY}/${PROJECT}/tomcat-${ARCH}:${VERSION}

clean:
	rm -rf bin build

.PHONY: setup build_test push_test list_test generate_images publish_images list_setup clean
