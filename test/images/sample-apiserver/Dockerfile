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

FROM k8s.gcr.io/kube-cross:v1.10.4-1 as build_k8s_1_10_sample_apiserver

ENV GOPATH /go
RUN mkdir -p ${GOPATH}/src ${GOPATH}/bin
ENV PATH $GOPATH/bin:$PATH

# The e2e aggregator test was originally added in #50347 and is designed to test ability to run a 1.7
# sample-apiserver in newer releases. please see e2e test named "Should be able to support the 1.7 Sample
# API Server using the current Aggregator"
RUN go get -d k8s.io/sample-apiserver \
    && cd ${GOPATH}/src/k8s.io/sample-apiserver \
    && git checkout --track origin/release-1.10 \
    && CGO_ENABLED=0 GOOS=linux GOARCH=BASEARCH go install .

# for arm, go install uses go/bin/linux_arm, so just find the file and copy it to the root so
# we can copy it out from this throw away container image from a standard location
RUN find /go/bin -name sample-apiserver -exec cp {} / \;

FROM BASEIMAGE
COPY --from=build_k8s_1_10_sample_apiserver /sample-apiserver /sample-apiserver

ENTRYPOINT ["/sample-apiserver"]
