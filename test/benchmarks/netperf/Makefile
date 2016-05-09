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

all: docker nptests push docker-client

# Unfortunately, we need to build two Docker containers - one with no ports exposed
# otherwise net=host mode cannot colocate W1 and W2 pods on the same node as needed
# for testing within node network performance

DOCKERREPO       := girishkalele/netperf-latest
DOCKERCLIENTREPO := girishkalele/netperf-client-latest

docker: nptests
	mkdir -p Dockerbuild && \
	cp -f Dockerfile Dockerbuild/ && \
	cp -f nptests Dockerbuild/ &&\
	docker build -t $(DOCKERCLIENTREPO) Dockerbuild/ 

docker-client: nptests docker
	mkdir -p Dockerbuildclient && \
	cp -f Dockerfile-client Dockerbuildclient/Dockerfile && \
	cp -f nptests Dockerbuildclient/ &&\
	docker build -t $(DOCKERREPO) Dockerbuildclient/ 

push: docker docker-client
	gcloud docker push $(DOCKERCLIENTREPO)
	gcloud docker push $(DOCKERREPO)

authorize_gcloud:
	gcloud docker --authorize-only

authorize:
	docker login

nptests: nptest.go
	go build -o nptests

clean:
	@rm -f Dockerbuild/*
	@rm -f Dockerbuildclient/*
	@rm -f ./nptests ./*svg ./*png ./*jpg ./*csv ./*.cap ./*.pyc nptests
	@go clean

# The list of Python library packages need by the launch script
PYTHON-DEP_LIBS := python-jinja2 python-matplotlib

installdeps:
	sudo apt-get install $(PYTHON_DEP_LIBS)

runtests:
	@echo Launching network performance tests
	./launch.py

