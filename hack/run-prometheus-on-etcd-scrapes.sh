#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# Unpacks a tarfile of etcd scrapes and runs a simple web server exposing it
# and a Prometheus server scraping that simple web server.
# The simple web server listens on port 9091.
# The Prometheus server is run in a container and looks for the
# simple web server at the host's first global IPv4 address.

# Usage: $0 scrapes_tar_pathname
#
# Where scrapes_tar_pathname is a gzipped tar archive containing
# files whose name is of the form
# <timestamp>.scrape
# where <timestamp> is seconds since Jan 1, 1970 UTC.
# Each such file is taken to be a scrape that lacks timestamps,
# and the timestamp from the filename is multiplied by the necessary 1000
# and added to the data in that file.

# This requires a:
# - `docker run` command
# - an `ip` or `ifconfig` command that this script knows how to wrangle
# - an `nc` command that serve-prom-scrapes.sh knows how to wrangle

if (( $# != 1 )); then
    echo "Usage: $0 \$scrapes_tar_pathname" >&2
    exit 1
fi

scrapes_file="$1"

if ! [[ -r "$scrapes_file" ]]; then
    echo "$0: $scrapes_file is not a readable file" >&2
    exit 2
fi

SCRIPT_ROOT=$(dirname "${BASH_SOURCE[0]}")

CONFIG="/tmp/$(cd /tmp && mktemp config.XXXXXX)"
UNPACKDIR="/tmp/$(cd /tmp && mktemp -d unpack.XXXXXX)"
SERVER_PID=""

cleanup_prom() {
    rm -f "$CONFIG"
    rm -rf "$UNPACKDIR"
    if [[ -n "$SERVER_PID" ]]; then
	kill "$SERVER_PID"
    fi
}

trap cleanup_prom EXIT

chmod +r "$CONFIG" "$UNPACKDIR"

tar xzf "$scrapes_file" -C "$UNPACKDIR"

if which ip > /dev/null; then
    IPADDR=$(ip addr show scope global up |
	     grep -w inet | head -1 |
	     awk '{ print $2 }' | awk -F/ '{ print $1 }')
else
    IPADDR=$(ifconfig | grep -w inet | grep -Fv 127.0.0. | head -1 |
	     awk '{ print $2 }' | awk -F/ '{ print $1 }')
fi

echo
echo "Historic metrics will be at http://\${any_local_address}:9091/\${any_path}"
echo "Prometheus will listen on port 9090 and scrape historic metrics from http://${IPADDR}:9091/metrics"
sleep 1
echo

cat > "$CONFIG" <<EOF
global:
  scrape_interval: 30s

scrape_configs:

- job_name: local
  static_configs:
  - targets: ['${IPADDR}:9091']
EOF

"${SCRIPT_ROOT}/serve-prom-scrapes.sh" 9091 "$UNPACKDIR" &
SERVER_PID=$!
docker run -p 9090:9090 -v "${CONFIG}:/config.yaml" prom/prometheus --config.file=/config.yaml --storage.tsdb.retention.time=3650d
