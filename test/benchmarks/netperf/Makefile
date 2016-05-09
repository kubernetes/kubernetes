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

all: docker nptests push launch runtests

DOCKERREPO       := girishkalele/netperf-latest

docker: nptests launch
	mkdir -p Dockerbuild && \
	cp -f Dockerfile Dockerbuild/ && \
	cp -f nptest/nptests Dockerbuild/ &&\
	docker build -t $(DOCKERREPO) Dockerbuild/ 

push: docker
	gcloud docker push $(DOCKERREPO)

nptests: nptest/nptest.go
	cd nptest; make nptests

clean:
	@rm -f Dockerbuild/*
	@rm -f ./nptests ./*svg ./*png ./*jpg ./*csv ./*.cap ./*.pyc nptests ./*~
	@go clean
	cd nptest; make clean

launch: launch.go
	go build -o launch launch.go

# 'runtests' is the test runner target
runtests: launch
	@echo Launching network performance tests
	./launch
	@echo Network performance tests complete - running matplotlib to generate graphs and copy them back
	cp netperf-latest.csv plotperf && cd plotperf; make plot && mv *png .. && mv *svg ..
	@echo Results file netperf-latest.csv and SVG/PNG graphs generated successfully


