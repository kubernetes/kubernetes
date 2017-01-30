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

# build the cassandra image.

VERSION=v11
PROJECT_ID=google_samples
PROJECT=gcr.io/${PROJECT_ID}

all: build

kubernetes-cassandra.jar: ../java/* ../java/src/main/java/io/k8s/cassandra/*.java
	cd ../java && mvn clean && mvn package
	mv ../java/target/kubernetes-cassandra*.jar files/kubernetes-cassandra.jar
	cd ../java && mvn clean

build: kubernetes-cassandra.jar
	docker build --pull -t ${PROJECT}/cassandra:${VERSION} .

push: build
	gcloud docker -- push ${PROJECT}/cassandra:${VERSION}

.PHONY: all build push
