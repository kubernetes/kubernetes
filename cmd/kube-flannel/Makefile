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
IMAGE = gcr.io/mikedanese-k8s/kube-flannel
TAG ?= dev

version=1.4.21
source=ftp://ftp.netfilter.org/pub/iptables/iptables-$(version).tar.bz2

default: container

xtables-multi:
	mkdir -p .tmp/
	curl --fail -q -sSL $(source) \
		| tar xjv --strip-components 1 -C .tmp/
	( cd .tmp/; ./configure --enable-static --disable-shared )
	$(MAKE) -C .tmp/ CFLAGS='-static' LDFLAGS='-static -dl'
	mv .tmp/iptables/xtables-multi .

build:
	go install ./...
	go build --ldflags '-extldflags "-static"' -o kube-flannel .
PHONY += build

container: build xtables-multi
	docker build -t $(IMAGE):$(TAG) .
PHONY += container

push: container
	docker push $(IMAGE):$(TAG)
PHONY += push

