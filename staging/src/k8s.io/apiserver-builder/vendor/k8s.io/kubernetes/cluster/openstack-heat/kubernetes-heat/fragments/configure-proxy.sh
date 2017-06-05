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

set -o errexit
set -o nounset
set -o pipefail

# The contents of these variables swapped in by heat via environments presented to kube-up.sh

export ETC_ENVIRONMENT='FTP_PROXY=$FTP_PROXY
HTTP_PROXY=$HTTP_PROXY
HTTPS_PROXY=$HTTPS_PROXY
SOCKS_PROXY=$SOCKS_PROXY
NO_PROXY=$NO_PROXY
ftp_proxy=$FTP_PROXY
http_proxy=$HTTP_PROXY
https_proxy=$HTTPS_PROXY
socks_proxy=$SOCKS_PROXY
no_proxy=$NO_PROXY
'

export ETC_PROFILE_D='export FTP_PROXY=$FTP_PROXY
export HTTP_PROXY=$HTTP_PROXY
export HTTPS_PROXY=$HTTPS_PROXY
export SOCKS_PROXY=$SOCKS_PROXY
export NO_PROXY=$NO_PROXY
export ftp_proxy=$FTP_PROXY
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
export socks_proxy=$SOCKS_PROXY
export no_proxy=$NO_PROXY
'

export DOCKER_PROXY='[Service]
      Environment="HTTP_PROXY=$HTTP_PROXY"
      Environment="HTTPS_PROXY=$HTTPS_PROXY"
      Environment="SOCKS_PROXY=$SOCKS_PROXY"
      Environment="NO_PROXY=$NO_PROXY"
      Environment="ftp_proxy=$FTP_PROXY"
      Environment="http_proxy=$HTTP_PROXY"
      Environment="https_proxy=$HTTPS_PROXY"
      Environment="socks_proxy=$SOCKS_PROXY"
      Environment="no_proxy=$NO_PROXY"
'

# This again is set by heat
ENABLE_PROXY='$ENABLE_PROXY'

# Heat itself doesn't have conditionals, so this is how we set up our proxy without breaking non-proxy setups.
if [[ "${ENABLE_PROXY}" == "true" ]]; then
  mkdir -p /etc/systemd/system/docker.service.d/

  echo "${ETC_ENVIRONMENT}" >> /etc/environment
  echo "${ETC_PROFILE_D}" > /etc/profile.d/proxy_config.sh
  echo "${DOCKER_PROXY}" > etc/systemd/system/docker.service.d/http-proxy.conf
  echo "proxy=$HTTP_PROXY" >> /etc/yum.conf
fi
