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

FROM docker.elastic.co/elasticsearch/elasticsearch:5.5.1

USER root

RUN mkdir /data
RUN chown -R elasticsearch:elasticsearch /data

WORKDIR /usr/share/elasticsearch

VOLUME ["/data"]
EXPOSE 9200 9300

USER elasticsearch
COPY elasticsearch_logging_discovery bin/
COPY config/elasticsearch.yml config/
COPY config/log4j2.properties config/
COPY run.sh bin/

USER root
RUN chown -R elasticsearch:elasticsearch config
CMD ["bin/run.sh"]
