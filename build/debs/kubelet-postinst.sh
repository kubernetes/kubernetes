#!/bin/sh

# Copyright 2018 The Kubernetes Authors.
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

if [ -x /bin/systemctl ] && systemctl is-active systemd-resolved --quiet ; then
  mkdir -p /etc/systemd/system/kubelet.service.d
  cat <<EOF > /etc/systemd/system/kubelet.service.d/09-systemd-resolved.conf
[Service]
Environment="KUBELET_RESOLVER_ARGS=--resolv-conf=/run/systemd/resolve/resolv.conf"
ExecStart=
ExecStart=/usr/bin/kubelet $KUBELET_RESOLVER_ARGS
EOF
fi
