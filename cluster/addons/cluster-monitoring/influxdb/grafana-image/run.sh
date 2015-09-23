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

HEADER_CONTENT_TYPE="Content-Type: application/json"
HEADER_ACCEPT="Accept: application/json"

GRAFANA_USER=${GRAFANA_USER:-admin}
GRAFANA_PASSWD=${GRAFANA_PASSWD:-admin}
GRAFANA_PORT=${GRAFANA_PORT:-3000}

INFLUXDB_HOST=${INFLUXDB_HOST:-"monitoring-influxdb"}
INFLUXDB_DATABASE=${INFLUXDB_DATABASE:-k8s}
INFLUXDB_PASSWORD=${INFLUXDB_PASSWORD:-root}
INFLUXDB_PORT=${INFLUXDB_PORT:-8086}
INFLUXDB_USER=${INFLUXDB_USER:-root}

DASHBOARD_LOCATION=${DASHBOARD_LOCATION:-"/dashboards"}

# Allow access to dashboards without having to log in
export GF_AUTH_ANONYMOUS_ENABLED=true
export GF_SERVER_HTTP_PORT=${GRAFANA_PORT}

BACKEND_ACCESS_MODE=${BACKEND_ACCESS_MODE:-proxy}
INFLUXDB_SERVICE_ENDPOINT=${INFLUXDB_SERVICE_ENDPOINT}
if [ -n "$INFLUXDB_SERVICE_ENDPOINT" ]; then
  echo "Influxdb endpoint is provided."
else
  echo "Discovering influxdb endpoint..."
  INFLUXDB_SERVICE_ENDPOINT=$(/influxdb_service_discovery)
  if [ -n "$INFLUXDB_SERVICE_ENDPOINT" ]; then
    echo "Use InfluxDB external endpoint, and 'direct' access mode from Grafana."
    BACKEND_ACCESS_MODE=direct
  else
    echo "Unable to get external service endpoint for InfluxDB."
    echo "Use internal endpoint, and 'proxy' access mode from Grafana."
    INFLUXDB_SERVICE_ENDPOINT="http://${INFLUXDB_HOST}:${INFLUXDB_PORT}"
    BACKEND_ACCESS_MODE=proxy
  fi
fi

echo "Using the following endpoint for InfluxDB: ${INFLUXDB_SERVICE_ENDPOINT}"
echo "Using the following backend access mode for InfluxDB: ${BACKEND_ACCESS_MODE}"

set -m
echo "Starting Grafana in the background"
exec /usr/sbin/grafana-server --config=/etc/grafana/grafana.ini cfg:default.paths.data=/var/lib/grafana cfg:default.paths.logs=/var/log/grafana &

echo "Waiting for Grafana to come up..."
until $(curl --fail --output /dev/null --silent http://${GRAFANA_USER}:${GRAFANA_PASSWD}@localhost:${GRAFANA_PORT}/api/org); do
  printf "."
  sleep 2
done
echo "Grafana is up and running."
echo "Creating default influxdb datasource..."
curl -i -XPOST -H "${HEADER_ACCEPT}" -H "${HEADER_CONTENT_TYPE}" "http://${GRAFANA_USER}:${GRAFANA_PASSWD}@localhost:${GRAFANA_PORT}/api/datasources" -d '
{ 
  "name": "influxdb-datasource",
  "type": "influxdb",
  "access": "'"${BACKEND_ACCESS_MODE}"'",
  "isDefault": true,
  "url": "'"${INFLUXDB_SERVICE_ENDPOINT}"'",
  "password": "'"${INFLUXDB_PASSWORD}"'",
  "user": "'"${INFLUXDB_USER}"'",
  "database": "'"${INFLUXDB_DATABASE}"'"
}'

echo ""
echo "Importing default dashboards..."
for filename in ${DASHBOARD_LOCATION}/*.json; do
  echo "Importing ${filename} ..."
  curl -i -XPOST --data "@${filename}" -H "${HEADER_ACCEPT}" -H "${HEADER_CONTENT_TYPE}" "http://${GRAFANA_USER}:${GRAFANA_PASSWD}@localhost:${GRAFANA_PORT}/api/dashboards/db"
  echo ""
  echo "Done importing ${filename}"
done
echo ""
echo "Bringing Grafana back to the foreground"
fg

