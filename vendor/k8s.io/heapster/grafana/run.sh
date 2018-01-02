#!/bin/bash

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
export GF_AUTH_ANONYMOUS_ENABLED=${GF_AUTH_ANONYMOUS_ENABLED:-true}
export GF_SERVER_HTTP_PORT=${GRAFANA_PORT}

GF_SERVER_PROTOCOL=${GF_SERVER_PROTOCOL:-http}

BACKEND_ACCESS_MODE=${BACKEND_ACCESS_MODE:-proxy}
INFLUXDB_SERVICE_URL=${INFLUXDB_SERVICE_URL}
if [ -n "$INFLUXDB_SERVICE_URL" ]; then
  echo "Influxdb service URL is provided."
else
  INFLUXDB_SERVICE_URL="http://${INFLUXDB_HOST}:${INFLUXDB_PORT}"
fi

echo "Using the following URL for InfluxDB: ${INFLUXDB_SERVICE_URL}"
echo "Using the following backend access mode for InfluxDB: ${BACKEND_ACCESS_MODE}"

set -m
echo "Starting Grafana in the background"
exec /usr/sbin/grafana-server --homepath=/usr/share/grafana --config=/etc/grafana/grafana.ini cfg:default.paths.data=/var/lib/grafana cfg:default.paths.logs=/var/log/grafana &

echo "Waiting for Grafana to come up..."
until $(curl -k --fail --output /dev/null --silent ${GF_SERVER_PROTOCOL}://${GRAFANA_USER}:${GRAFANA_PASSWD}@localhost:${GRAFANA_PORT}/api/org); do
  printf "."
  sleep 2
done
echo "Grafana is up and running."
echo "Creating default influxdb datasource..."
curl -k -i -XPOST -H "${HEADER_ACCEPT}" -H "${HEADER_CONTENT_TYPE}" "${GF_SERVER_PROTOCOL}://${GRAFANA_USER}:${GRAFANA_PASSWD}@localhost:${GRAFANA_PORT}/api/datasources" -d '
{
  "name": "influxdb-datasource",
  "type": "influxdb",
  "access": "'"${BACKEND_ACCESS_MODE}"'",
  "isDefault": true,
  "url": "'"${INFLUXDB_SERVICE_URL}"'",
  "password": "'"${INFLUXDB_PASSWORD}"'",
  "user": "'"${INFLUXDB_USER}"'",
  "database": "'"${INFLUXDB_DATABASE}"'"
}'

echo ""
echo "Importing default dashboards..."
for filename in ${DASHBOARD_LOCATION}/*.json; do
  echo "Importing ${filename} ..."
  curl -k -i -XPOST --data "@${filename}" -H "${HEADER_ACCEPT}" -H "${HEADER_CONTENT_TYPE}" "${GF_SERVER_PROTOCOL}://${GRAFANA_USER}:${GRAFANA_PASSWD}@localhost:${GRAFANA_PORT}/api/dashboards/db"
  echo ""
  echo "Done importing ${filename}"
done
echo ""
echo "Bringing Grafana back to the foreground"
fg

