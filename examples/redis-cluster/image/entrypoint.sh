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

set -e

if [ -n "${CLUSTER_CONFIG}" ]; then

  printf "Applying cluster configuration from CLUSTER_CONFIG:\n"

  while read -r line
  do
    printf "$line\n" >> /tmp/nodes.tmp
  done <<< "$CLUSTER_CONFIG"

  while read -r line
  do
    eval echo $line
    eval echo $line >> nodes.conf
  done < /tmp/nodes.tmp
  
fi

if [ -n "${REDIS_CONFIG}" ]; then

  printf "Applying redis configuration from REDIS_CONFIG:\n"

  while read -r line
  do
    printf "$line\n" >> /tmp/redis.tmp
  done <<< "$REDIS_CONFIG"

  while read -r line
  do
    eval echo $line
    eval echo $line >> /etc/redis.conf
  done < /tmp/redis.tmp

  REDIS_CONFIG_FILE=/etc/redis.conf
fi

if [ "$1" = 'redis-server' ]; then
  chown -R redis .
  shift
  exec gosu redis redis-server $REDIS_CONFIG_FILE "$@"
fi

exec "$@"