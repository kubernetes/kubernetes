#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
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

set -euo pipefail

local_port=${1:-3000}
[[ $local_port =~ ^[0-9]+$ ]] && ((local_port >= 1 && local_port <= 65535)) || {
  echo "usage: $0 [local-port] (1-65535)" >&2
  exit 2
}

ssh_key=${WATCHLIST_SSH_KEY:-"$HOME/.ssh/id_rsa"}
remote_host=${WATCHLIST_REMOTE_HOST:-lukasz.szaszkiewicz@34.118.97.146}
cluster_name=${WATCHLIST_CLUSTER_NAME:-watchlist-branch}
[[ -f $ssh_key ]] || { echo "SSH key not found: $ssh_key" >&2; exit 1; }

ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$ssh_key" "$remote_host" \
  "WATCHLIST_CLUSTER_NAME=$(printf '%q' "$cluster_name") bash ~/perf-tests/clusterloader2/testing/list/watchlist-profile-remote.sh grafana"

echo "Dashboard: http://127.0.0.1:$local_port/d/watchlist-remote/watchlist-remote-benchmark" >&2
echo "Grafana: http://127.0.0.1:$local_port (keep this tunnel open)" >&2
exec ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -i "$ssh_key" -L "127.0.0.1:$local_port:127.0.0.1:3000" "$remote_host"
