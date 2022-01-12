#!/bin/bash

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

# These steps must be executed once the host /var and /lib volumes have
# been mounted, and therefore cannot be done in the docker build stage.

# For systems without journald
mkdir -p /var/log/journal

# set ld preload
if dpkg --print-architecture | grep -q amd64;then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
else
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
fi

# For disabling elasticsearch ruby client sniffering feature.
# Because, on k8s, sniffering feature sometimes causes failed to flush buffers error
# due to between service name and ip address glitch.
# And this should be needed for downstream helm chart configurations
# for sniffer_class_name parameter.
SIMPLE_SNIFFER=$( gem contents fluent-plugin-elasticsearch | grep elasticsearch_simple_sniffer.rb )
if [ -n "$SIMPLE_SNIFFER" ] && [ -f "$SIMPLE_SNIFFER" ] ; then
    FLUENTD_ARGS="$FLUENTD_ARGS -r $SIMPLE_SNIFFER"
fi

# Use exec to get the signal
# A non-quoted string and add the comment to prevent shellcheck failures on this line.
# See https://github.com/koalaman/shellcheck/wiki/SC2086
# shellcheck disable=SC2086
exec /usr/local/bundle/bin/fluentd $FLUENTD_ARGS
