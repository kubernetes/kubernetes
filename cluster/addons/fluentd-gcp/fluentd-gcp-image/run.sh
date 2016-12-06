#!/bin/sh

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

# For systems without journald
mkdir -p /var/log/journal

if [ -e /host/lib/libsystemd* ]
then
  rm /lib/x86_64-linux-gnu/libsystemd*
  cp /host/lib/libsystemd* /lib/x86_64-linux-gnu/
fi

LD_PRELOAD=/opt/td-agent/embedded/lib/libjemalloc.so
RUBY_GC_HEAP_OLDOBJECT_LIMIT_FACTOR=0.9

/usr/sbin/td-agent $@
