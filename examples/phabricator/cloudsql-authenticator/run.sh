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

# TODO: This loop updates authorized networks even if nothing has changed. It
#       should only send updates if something changes. We should be able to do
#       this by comparing pod creation time with the last scan time.
while true; do
  hostport="${KUBERNETES_RO_SERVICE_HOST}:${KUBERNETES_RO_SERVICE_PORT}"
  path="api/v1beta1/pods"
  query="labels=$SELECTOR"
  ips_json=`curl ${hostport}/${path}?${query} 2>/dev/null | grep hostIP`
  ips=`echo $ips_json | cut -d'"' -f 4 | sed 's/,$//'`
  echo "Adding IPs $ips"
  gcloud sql instances patch $CLOUDSQL_DB --authorized-networks $ips
  sleep 10
done
