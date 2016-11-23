#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

echo "If you get errors below, kubernetes env injection could be failing..."
echo "env vars ="
env
echo "CHECKING ENVS BEFORE STARTUP........"
if [ ! "$REDISMASTER_SERVICE_HOST" ]; then
    echo "Need to set REDIS_MASTER_SERVICE_HOST" && exit 1;
fi
if [ ! "$REDISMASTER_PORT" ]; then
    echo "Need to set REDIS_MASTER_PORT" && exit 1;
fi

echo "ENV Vars look good, starting !"

redis-server --slaveof ${REDISMASTER_SERVICE_HOST:-$SERVICE_HOST} $REDISMASTER_SERVICE_PORT
