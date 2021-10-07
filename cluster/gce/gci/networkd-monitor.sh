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

# This script is for master and node instance health monitoring, which is
# packed in kube-manifest tarball. It is executed through a systemd service
# in cluster/gce/gci/<master/node>.yaml. The env variables come from an env
# file provided by the systemd service.

while true; do
  2>&1 systemctl status systemd-networkd>/dev/null
  if [[ $? -ne 0 ]]; then
    echo "$(date) systemd-networkd has stopped, restarting..."
    echo "Dumping journalctl logs systemd-networkd..."
    journalctl --since="-1d" --no-pager -u systemd-networkd --dir=/var/log/journal
    echo "Now running systemctl restart systemd-networkd..."
    systemctl restart systemd-networkd
  fi
  sleep 60
done
