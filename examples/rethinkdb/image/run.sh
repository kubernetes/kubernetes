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

set -o pipefail

IP=""
if [[ -n "${KUBERNETES_RO_SERVICE_HOST}" ]]; then

  : ${NAMESPACE:=rethinkdb}
  # try to pick up first different ip from endpoints
  MYHOST=$(ip addr | grep 'state UP' -A2 | tail -n1 | awk '{print $2}' | cut -f1  -d'/')
  URL="${KUBERNETES_RO_SERVICE_HOST}/api/v1beta3/namespaces/${NAMESPACE}/endpoints/rethinkdb-driver"
  IP=$(curl -s ${URL} | jq -s -r --arg h "${MYHOST}" '.[0].subsets | .[].addresses | [ .[].IP ] | map(select(. != $h)) | .[0]') || exit 1
  [[ "${IP}" == null ]] && IP=""
fi

if [[ -n "${IP}" ]]; then
  ENDPOINT="${IP}:29015"
  echo "Join to ${ENDPOINT}"
  exec rethinkdb --bind all  --join ${ENDPOINT}
else
  echo "Start single instance"
  exec rethinkdb --bind all
fi
