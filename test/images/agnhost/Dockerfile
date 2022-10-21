# Copyright 2019 The Kubernetes Authors.
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
FROM $BASEIMAGE

CROSS_BUILD_COPY qemu-QEMUARCH-static /usr/bin/

# from dnsutils image
# install necessary packages:
# - bind-tools: contains dig, which can used in DNS tests.
# - CoreDNS: used in some DNS tests.
# from hostexec image
# install necessary packages:
# - curl, nc: used by a lot of e2e tests
# - iproute2: includes ss used in NodePort tests
# from iperf image
# install necessary packages: iperf, bash
RUN apk --update add bind-tools curl netcat-openbsd iproute2 iperf bash && rm -rf /var/cache/apk/* \
  && ln -s /usr/bin/iperf /usr/local/bin/iperf \
  && ls -altrh /usr/local/bin/iperf

ADD https://github.com/coredns/coredns/releases/download/v1.6.2/coredns_1.6.2_linux_BASEARCH.tgz /coredns.tgz
RUN tar -xzvf /coredns.tgz && rm -f /coredns.tgz

# PORT 80 needed by: test-webserver
# PORT 8080 needed by: netexec, nettest, resource-consumer, resource-consumer-controller
# PORT 8081 needed by: netexec
# PORT 9376 needed by: serve-hostname
# PORT 5000 needed by: grpc-health-checking
EXPOSE 80 8080 8081 9376 5000

# from netexec
RUN mkdir /uploads

# from porter
ADD porter/localhost.crt localhost.crt
ADD porter/localhost.key localhost.key

ADD agnhost agnhost

# needed for the entrypoint-tester related tests. Some of the entrypoint-tester related tests
# overrides this image's entrypoint with agnhost-2 binary, and will verify that the correct
# entrypoint is used by the containers.
RUN ln -s agnhost agnhost-2

# this user and group is used in a E2E test case of 
# SupplementalGroups with pre-defined group in the image
# - user-defined-in-image(uid=1000)
# - user-defined-in-image belongs to group-defined-in-image(gid=50000)
RUN adduser -u 1000 -D user-defined-in-image && \
    addgroup -g 50000 group-defined-in-image && \
    addgroup user-defined-in-image group-defined-in-image

ENTRYPOINT ["/agnhost"]
CMD ["pause"]
