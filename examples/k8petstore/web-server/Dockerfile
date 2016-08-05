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

FROM golang:latest

# Add source to gopath.  This is defacto required for go apps.
ADD ./src /gopath/src/k8petstore
RUN mkdir /gopath/bin/
ADD ./static /tmp/static
ADD ./test.sh /opt/test.sh
RUN chmod 777 /opt/test.sh

# So that we can easily run and install
WORKDIR /gopath/src

# Install the code (the executables are in the main dir)  This will get the deps also.
RUN export GOPATH=/gopath/ && go get k8petstore
RUN export GOPATH=/gopath/ && go install k8petstore


# Expected that you will override this in production kubernetes.
ENV STATIC_FILES /tmp/static
CMD /gopath/bin/k8petstore
