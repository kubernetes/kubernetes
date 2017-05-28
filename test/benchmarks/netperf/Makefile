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

all: docker nptests push

DOCKERREPO       := girishkalele/netperf-latest

docker: nptests
	mkdir -p Dockerbuild && \
	cp -f Dockerfile Dockerbuild/ && \
	cp -f nptests Dockerbuild/ &&\
	docker build -t $(DOCKERREPO) Dockerbuild/ 

push: docker
	gcloud docker push $(DOCKERREPO)

authorize_gcloud:
	gcloud docker --authorize-only

authorize:
	docker login

nptests: nptest.go
	go build -o nptests

clean:
	@rm -f Dockerbuild/*
	@rm -f ./nptests ./*svg ./*png ./*jpg ./*csv ./*.cap ./*.pyc nptests
	@go clean

# The list of Python library packages need by the launch script
PYTHON-DEP_LIBS := python-jinja2 python-matplotlib

installdeps:
	sudo apt-get install $(PYTHON_DEP_LIBS)

runtests:
	@echo Launching network performance tests
	./launch.py

