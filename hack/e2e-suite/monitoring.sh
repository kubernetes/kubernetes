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

MONITORING="${KUBE_ROOT}/cluster/addons/cluster-monitoring"
KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
BIGRAND=$(printf "%x\n" $(( $RANDOM << 16 | $RANDOM ))) # random 2^32 in hex
MONITORING_FIREWALL_RULE="monitoring-test-${BIGRAND}"

function setup {
  # This only has work to do on gce and gke
  if [[ "${KUBERNETES_PROVIDER}" == "gce" ]] || [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
    detect-project
    if ! "${GCLOUD}" compute firewall-rules create "${MONITORING_FIREWALL_RULE}" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --quiet \
      --allow tcp:80 tcp:8083 tcp:8086 tcp:9200; then
      echo "Failed to set up firewall for monitoring" && false
    fi
  fi

  "${KUBECTL}" create -f "${MONITORING}/"
}

function cleanup {
  "${KUBECTL}" stop rc monitoring-influx-grafana-controller &> /dev/null || true
  "${KUBECTL}" stop rc monitoring-heapster-controller &> /dev/null || true
r "${KUBECTL}" delete -f "${MONITORING}/" &> /dev/null || true

  # This only has work to do on gce and gke
  if [[ "${KUBERNETES_PROVIDER}" == "gce" ]] || [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
    detect-project
    if "${GCLOUD}" compute firewall-rules describe "${MONITORING_FIREWALL_RULE}" &> /dev/null; then
      "${GCLOUD}" compute firewall-rules delete \
        --project "${PROJECT}" \
        --quiet \
        "${MONITORING_FIREWALL_RULE}" || true
    fi
  fi
}

function influx-data-exists {
  local max_retries=10
  local retry_delay=30 #seconds
  local influx_ip=$("${KUBECTL}" get pods -l name=influxGrafana -o template -t {{range.items}}{{.currentState.hostIP}}:{{end}} | sed s/://g)
  local influx_url="http://$influx_ip:8086/db/k8s/series?u=root&p=root"
  local ok="false"
  for i in `seq 1 10`; do
    if curl --retry $max_retries --retry-delay $retry_delay -G $influx_url --data-urlencode "q=select * from stats limit 1" \
      && curl --retry $max_retries --retry-delay $retry_delay -G $influx_url --data-urlencode "q=select * from machine limit 1"; then
      echo "retrieved data from InfluxDB."
      ok="true"
      break
    fi
    sleep 5
  done
  if [[ "${ok}" != "true" ]]; then
    echo "failed to retrieve stats from InfluxDB. monitoring test failed"
    exit 1
  fi
}

function wait-for-pods {
  local running=false
  for i in `seq 1 20`; do
    sleep 20
    if "${KUBECTL}" get pods -l name=influxGrafana -o template -t {{range.items}}{{.currentState.status}}:{{end}} | grep Running &> /dev/null \
      && "${KUBECTL}" get pods -l name=heapster -o template -t {{range.items}}{{.currentState.status}}:{{end}} | grep Running &> /dev/null; then
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

# Remove any pre-existing monitoring services.
cleanup 

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
