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

FROM golang:1.8.0

RUN apt-get update; apt-get install git
RUN mkdir -p src/k8s.io/
WORKDIR /go/src/k8s.io
RUN git clone https://github.com/kubernetes/minikube
WORKDIR /go/src/k8s.io/minikube
RUN go build cmd/localkube/main.go
RUN apt-get install -y apt-utils nano less

# Add Docs stuff

RUN curl -sL https://deb.nodesource.com/setup_7.x | bash -
RUN apt-get install -y nodejs

RUN apt-get install -y build-essential
RUN mkdir -p /go/src/github.com/Birdrock
WORKDIR /go/src/github.com/Birdrock
RUN git clone --depth=1 https://github.com/Birdrock/brodocs.git
WORKDIR /go/src/github.com/Birdrock/brodocs
RUN npm install
#RUN node brodoc.js

RUN mkdir -p /go/src/github.com/pwittrock/apiserver-helloworld
ADD apiserver-helloworld/ /go/src/github.com/pwittrock/apiserver-helloworld/

WORKDIR /go/src/github.com/pwittrock/apiserver-helloworld/
ENV GOPATH /go/
RUN go build cmd/kubec/kubec.go
ENTRYPOINT ["./kubec"]
CMD ["init"]

#./kubec generate-docs --repo-name github.com/pwittrock/test
