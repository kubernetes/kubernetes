#!/bin/sh

# Copyright 2015 The Kubernetes Authors All rights reserved.
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


# Usage: $0 <protocol> <port> <service> [timeout]
#   protocol: tcp|udp - case insensitive
#   port: port number on which to receive and connect
#   service: the destination service name or IP
#   timeout: idle timeout in seconds (optional)

if [[ -z "$1" -o -z "$2" -o -z "$3" ]]; then
    echo "usage: $0 <protocol> <port> <service> [timeout]"
    exit 1
fi

PROTO=$(echo $1 | tr a-z A-Z)
TIMEOUT=""
test -n "$4" && TIMEOUT="-T$4"

CMD="socat ${TIMEOUT} ${PROTO}-LISTEN:$2,reuseaddr,fork ${PROTO}:$3:$2"
echo "Running ${CMD}"
exec ${CMD}
