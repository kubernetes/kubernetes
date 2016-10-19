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

# A Dockerfile for creating a Kibana container that is designed
# to work with Kubernetes logging.

FROM gcr.io/google_containers/ubuntu-slim:0.4

MAINTAINER Mik Vyatskov "vmik@google.com"

ENV DEBIAN_FRONTEND noninteractive
ENV KIBANA_VERSION 4.6.1

RUN apt-get update \
    && apt-get install -y curl \
    && apt-get clean

RUN set -x \
    && cd / \
    && mkdir /kibana \
    && curl -O https://download.elastic.co/kibana/kibana/kibana-$KIBANA_VERSION-linux-x86_64.tar.gz \
    && tar xf kibana-$KIBANA_VERSION-linux-x86_64.tar.gz -C /kibana --strip-components=1 \
    && rm kibana-$KIBANA_VERSION-linux-x86_64.tar.gz

COPY run.sh /run.sh

EXPOSE 5601

CMD ["/run.sh"]
