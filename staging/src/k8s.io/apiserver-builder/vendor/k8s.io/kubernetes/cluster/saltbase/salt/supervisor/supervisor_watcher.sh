#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# This script is invoked by crond every minute to check if supervisord is
# up and oom protected. If down it restarts supervisord; otherwise, it exits
# after applying oom_score_adj

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if ! /etc/init.d/supervisor status > /dev/null; then
    service supervisor start
    sleep 10
fi

# Apply oom_score_adj: -901 to processes
pids=$(cat /var/run/supervisord.pid)
for pid in "${pids}"; do
    echo -901 > /proc/$pid/oom_score_adj
done


