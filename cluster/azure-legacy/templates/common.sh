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

# Retry a download until we get it.
#
# $1 is the URL to download
download-or-bust() {
  local -r url="$1"
  local -r file="${url##*/}"
  rm -f "$file"
  until [[ -e "${file}" ]]; do
    curl --ipv4 -Lo "$file" --connect-timeout 20 --retry 6 --retry-delay 10 "$url"
    md5sum "$file"
  done
}

# Install salt from GCS.  See README.md for instructions on how to update these
# debs.
#
# $1 If set to --master, also install the master
install-salt() {
  apt-get update

  mkdir -p /var/cache/salt-install
  cd /var/cache/salt-install

  TARS=(
    libzmq3_3.2.3+dfsg-1~bpo70~dst+1_amd64.deb
    python-zmq_13.1.0-1~bpo70~dst+1_amd64.deb
    salt-common_2014.1.13+ds-1~bpo70+1_all.deb
    salt-minion_2014.1.13+ds-1~bpo70+1_all.deb
  )
  if [[ ${1-} == '--master' ]]; then
    TARS+=(salt-master_2014.1.13+ds-1~bpo70+1_all.deb)
  fi
  URL_BASE="https://storage.googleapis.com/kubernetes-release/salt"

  for tar in "${TARS[@]}"; do
    download-or-bust "${URL_BASE}/${tar}"
    dpkg -i "${tar}"
  done

  # This will install any of the unmet dependencies from above.
  apt-get install -f -y
}
