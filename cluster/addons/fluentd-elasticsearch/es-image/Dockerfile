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

FROM golang:1.16.6 AS builder
COPY elasticsearch_logging_discovery.go go.mod go.sum /
RUN CGO_ENABLED=0 GOOS=linux GO111MODULE=on go build -a -ldflags "-w" -o /elasticsearch_logging_discovery /elasticsearch_logging_discovery.go


FROM docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2

VOLUME ["/data"]
EXPOSE 9200 9300

COPY --from=builder /elasticsearch_logging_discovery bin/
COPY run.sh bin/
COPY config/elasticsearch.yml config/log4j2.properties config/

USER root
RUN chown -R elasticsearch:elasticsearch ./
CMD ["bin/run.sh"]
