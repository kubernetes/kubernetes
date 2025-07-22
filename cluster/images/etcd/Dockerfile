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

ARG BASEIMAGE
ARG RUNNERIMAGE

FROM ${BASEIMAGE} as builder

# This image needs bash for running "migrate-if-needed.sh". Instead of a full debian image
# we use just the bash-static and we wrap bash-static into a distroless image instead of
# a full debian image
RUN apt-get update -y \
	&& apt-get -yy -q install --no-install-recommends --no-install-suggests --fix-missing \
		bash-static

RUN cp /bin/bash-static /sh

FROM ${RUNNERIMAGE}
WORKDIR /

COPY --from=builder /sh /bin/

EXPOSE 2379 2380 4001 7001
# etcdctl is used by etcd.manifest for livenessProbe.
COPY etcd* etcdctl* /usr/local/bin/
COPY migrate-if-needed.sh migrate /usr/local/bin/
