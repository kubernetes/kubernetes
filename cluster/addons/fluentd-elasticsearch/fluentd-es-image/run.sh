#!/bin/bash

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

ES_HOST=${ES_HOST:-'elasticsearch-logging'};
ES_PORT=${ES_PORT:-9200};
ES_INDEX_NAME=${ES_INDEX_NAME:-'fluentd'};

[ -e /etc/td-agent/conf.d/output-elasticsearch.conf ] && {
    sed -i -r "s/%%ES_HOST%%/${ES_HOST}/g;s/%%ES_PORT%%/${ES_PORT}/g;s/%%ES_INDEX_NAME/${ES_INDEX_NAME}/g;" /etc/td-agent/conf.d/output-elasticsearch.conf
}
exec td-agent $@
