# Copyright 2018 The Kubernetes Authors.
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

ARG BASEIMAGE
FROM us.gcr.io/k8s-artifacts-prod/build-image/kube-cross:v1.13.9-5 as build_k8s_1_17_sample_apiserver

ENV GOPATH /go
RUN mkdir -p ${GOPATH}/src ${GOPATH}/bin
ENV PATH $GOPATH/bin:$PATH


# The e2e aggregator test is designed to test ability to run sample-apiserver as an aggregated server.
# see e2e test named "Should be able to support the 1.17 Sample API Server using the current Aggregator"

# Build v1.17.0 to ensure the current release supports a prior version of the sample apiserver
# Get without building to populate module cache
RUN GO111MODULE=on go get -d k8s.io/sample-apiserver@v0.17.0
# Get with OS/ARCH-specific env to build
RUN GO111MODULE=on CGO_ENABLED=0 GOOS=linux GOARCH=BASEARCH go get k8s.io/sample-apiserver@v0.17.0

# for arm, go install uses go/bin/linux_arm, so just find the file and copy it to the root so
# we can copy it out from this throw away container image from a standard location
RUN find /go/bin -name sample-apiserver -exec cp {} / \;

FROM $BASEIMAGE
COPY --from=build_k8s_1_17_sample_apiserver /sample-apiserver /sample-apiserver

ENTRYPOINT ["/sample-apiserver"]
