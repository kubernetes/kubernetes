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

# Serves a collection of scrape files up to Prometheus scraping.

# Usage: $0 port_num scrapes-dir
#
# Where scrapes-dir has descendant files whose name is of the form
# <timestamp>.scrape
# where <timestamp> is seconds since Jan 1, 1970 UTC.
# Each such file is taken to be a scrape that lacks timestamps,
# and the timestamp from the filename is multiplied by the necessary 1000
# and added to the data in that file.

# This requires an `nc` command that this script knows how to wrangle.

if (( $# != 2 )); then
    echo "Usage: $0 port_num scrapes_dir" >&2
    exit 1
fi

port_num="$1"
scrapes_dir="$2"
response_file="/tmp/$(cd /tmp && mktemp  response.XXXXXX)"

cleanup_serve() {
    rm -rf "$response_file"
}

trap cleanup_serve EXIT

chmod +r "$response_file"

transform() {
    path="$1"
    base="$(basename "$path")"
    seconds="${base%.scrape}"
    sed 's/^\([^#].*\)$/\1 '"${seconds}000/" "$path"
}

find_and_transform() {
    echo -n $'HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\n' > "$response_file"
    find "$scrapes_dir" -name "*.scrape" -print0 | sort -z | while read -d '' -r scrapename; do transform "$scrapename" >> "$response_file"; done
}

find_and_transform

if man nc | grep -wq -e -N
then dashen=-N
else dashen=
fi

# shellcheck disable=SC2086
while true; do nc -l $dashen 0.0.0.0 "$port_num" < "$response_file" > /dev/null; sleep 10; done
