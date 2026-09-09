#!/usr/bin/env bash
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

set -o errexit
set -o nounset
set -o pipefail

kubectl create -f conformance-e2e.yaml
while true; do
  STATUS=$(kubectl -n conformance get pods e2e-conformance-test -o jsonpath="{.status.phase}")
  timestamp=$(date +"[%H:%M:%S]")
  echo "$timestamp Pod status is: ${STATUS}"
  if [[ "$STATUS" == "Succeeded" ]]; then
    echo "$timestamp Done."
    break
  elif [[ "$STATUS" == "Failed" ]]; then
    echo "$timestamp Failed."
    kubectl -n conformance describe pods e2e-conformance-test || true
    kubectl -n conformance logs e2e-conformance-test || true
    exit 1
  else
    sleep 5
  fi
done
echo "Please use 'kubectl logs -n conformance e2e-conformance-test' to view the results"
