#!/bin/sh

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

if [ "${GET_HOSTS_FROM:-dns}" = "env" ]; then
  cat << EOF >> /etc/redis.conf
slaveof ${REDIS_MASTER_SERVICE_HOST} 6379
EOF
else
  cat << EOF >> /etc/redis.conf
slaveof redis-master 6379
EOF
fi

redis-server "/etc/redis.conf"
