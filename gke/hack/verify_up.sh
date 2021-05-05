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

set -e
set -u
set -o pipefail

kubectl version

# wait up to five minutes for all pods to be ready (30 attempts * 10 second delay)
delay=10
attempts=30
ready=false
for attempt in $(seq 1 $attempts)
do
    error_code=0
    wait_output=$(kubectl wait --for=condition=Ready pods --all -n kube-system --timeout=0 2>&1) || error_code=$?
    if [ "${error_code}" -eq 0 ]; then
        echo "$(date -Iseconds): PASS: all kube-system pods ready"
        echo "${wait_output}"
        ready=true
        break
    elif [ "${attempt}" -eq "${attempts}" ]; then
        echo "$(date -Iseconds): FAIL: not all kube-system pods ready after ${attempts} attempts"
        echo "${wait_output}"
        break
    else
        echo "$(date -Iseconds): waiting for kube-system pods to be ready (attempt ${attempt})..."
        sleep "${delay}"
    fi
done

# dump objects for visibility, tolerating timeouts and failures so we get the full output
# cluster state, lease/leader objects, workloads and pods
kubectl get cs,csr,nodes,lease,endpoints,all -A --request-timeout=10s || true
# events
kubectl get events -A -o wide --sort-by metadata.creationTimestamp --request-timeout=10s || true

if [ "${ready}" = true ]; then
    exit 0
else
    exit 1
fi

