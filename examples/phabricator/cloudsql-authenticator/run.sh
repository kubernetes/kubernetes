#!/bin/bash

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
