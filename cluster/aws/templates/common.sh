#!/bin/bash

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

# Retry a download until we get it.
#
# $1 is the URL to download
download-or-bust() {
  local -r url="$1"
  local -r file="${url##*/}"
  rm -f "$file"
  until [[ -e "${1##*/}" ]]; do
    echo "Downloading file ($1)"
    curl --ipv4 -Lo "$file" --connect-timeout 20 --retry 6 --retry-delay 10 "$1"
    md5sum "$file"
  done
}
