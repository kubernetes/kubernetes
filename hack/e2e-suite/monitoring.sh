#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# Assumes a running Kubernetes test cluster; verifies that the monitoring setup
# works. Assumes that we're being called by hack/e2e-test.sh (we use some env
# vars it sets up).

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/$KUBERNETES_PROVIDER/util.sh"

if [[ "${KUBERNETES_PROVIDER}" != "gce" ]] && [[ "${KUBERNETES_PROVIDER}" != "gke" ]]; then
  echo "WARNING: Skipping monitoring.sh for cloud provider: ${KUBERNETES_PROVIDER}."
  exit 0
fi

MONITORING="${KUBE_ROOT}/examples/monitoring"
KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
MONITORING_FIREWALL_RULE="monitoring-test"

function setup {
  detect-project

  if ! "${GCLOUD}" compute firewall-rules describe $MONITORING_FIREWALL_RULE &> /dev/null; then
    if ! "${GCLOUD}" compute firewall-rules create $MONITORING_FIREWALL_RULE \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --quiet \
      --allow tcp:80 tcp:8083 tcp:8086 tcp:9200; then
      echo "Failed to set up firewall for monitoring" && false
    fi
  fi

  "${KUBECTL}" create -f "${MONITORING}/influx-grafana-pod.json"
  "${KUBECTL}" create -f "${MONITORING}/influx-grafana-service.json"
  "${KUBECTL}" create -f "${MONITORING}/heapster-pod.json"
}

function cleanup {
  detect-project
  "${KUBECTL}" delete -f "${MONITORING}/influx-grafana-pod.json" || true
  "${KUBECTL}" delete -f "${MONITORING}/influx-grafana-service.json" || true
  "${KUBECTL}" delete -f "${MONITORING}/heapster-pod.json" || true
  if "${GCLOUD}" compute firewall-rules describe $MONITORING_FIREWALL_RULE &> /dev/null; then
    "${GCLOUD}" compute firewall-rules delete \
      --project "${PROJECT}" \
      --quiet \
      $MONITORING_FIREWALL_RULE || true
  fi
}

function influx-data-exists {
  local influx_ip=$("${KUBECTL}" get -o json pods influx-grafana | grep hostIP | awk '{print $2}' | sed 's/["|,]//g')
  local influx_url="http://$influx_ip:8086/db/k8s/series?u=root&p=root"
  if ! curl -G $influx_url --data-urlencode "q=select * from stats limit 1" \
    || ! curl -G $influx_url --data-urlencode "q=select * from machine limit 1"; then
    echo "failed to retrieve stats from Infludb. monitoring test failed"
    exit 1
  fi
}

function wait-for-pods {
  local running=false
  for i in `seq 1 20`; do
    sleep 20
    if "${KUBECTL}" get pods influx-grafana | grep Running &> /dev/null \
      && "${KUBECTL}" get pods heapster | grep Running &> /dev/null; then
      running=true
      break
    fi
  done
  if [ running == false ]; then
      echo "giving up waiting on monitoring pods to be active. monitoring test failed"
      exit 1
  fi
}

trap cleanup EXIT

# Start monitoring pods and services.
setup

# Wait for a maximum of 5 minutes for the influx grafana pod to be running.
echo "waiting for monitoring pods to be running"
wait-for-pods

# Wait for some time to let heapster push some stats to InfluxDB.
echo "monitoring pods are running. waiting for stats to be pushed to InfluxDB"
sleep 60

# Check if stats data exists in InfluxDB
echo "checking if stats exist in InfluxDB"
influx-data-exists

echo "monitoring setup works"
exit 0

