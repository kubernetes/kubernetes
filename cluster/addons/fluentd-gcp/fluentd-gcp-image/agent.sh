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

# For systems without systemd
mkdir -p /var/log/journal

export GEM_HOME="/opt/ruby/lib/ruby/gems/2.3.0/"
export GEM_PATH="/opt/ruby/lib/ruby/gems/2.3.0/"
export FLUENT_CONF="/etc/fluent/fluent.conf"
export FLUENT_SOCKET="/etc/fluent/agent.sock"

/opt/ruby/bin/fluentd "$@"
