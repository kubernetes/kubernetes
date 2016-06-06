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

CFG=/opt/redis/redis.conf
HOSTNAME=$(hostname)
DATADIR="/data"
# Port on which redis listens for connections.
PORT=6379

# Ping everyone but ourself to see if there's a master. Only one pet starts at
# a time, so if we don't see a master we can assume the position is ours.
while read -ra LINE; do
    if [[ "${LINE}" == *"${HOSTNAME}"* ]]; then
        sed -i -e "s|^bind.*$|bind ${LINE}|" ${CFG}
    elif [ "$(/opt/redis/redis-cli -h $LINE info | grep role | sed 's,\r$,,')" = "role:master" ]; then
        # TODO: More restrictive regex?
        sed -i -e "s|^.*slaveof.*$|slaveof ${LINE} ${PORT}|" ${CFG}
    fi
done

# Set the data directory for append only log and snapshot files. This should
# be a persistent volume for consistency.
sed -i -e "s|^.*dir .*$|dir ${DATADIR}|" ${CFG}

# The append only log is written for every SET operation. Without this setting,
# redis just snapshots periodically which is only safe for a cache. This will
# produce an appendonly.aof file in the configured data dir.
sed -i -e "s|^appendonly .*$|appendonly yes|" ${CFG}

# Every write triggers an fsync. Recommended default is "everysec", which
# is only safe for AP applications.
sed -i -e "s|^appendfsync .*$|appendfsync always|" ${CFG}


