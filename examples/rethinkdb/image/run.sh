#!/bin/bash

set -o pipefail

IP=""
if [[ -n "${KUBERNETES_RO_SERVICE_HOST}" ]]; then

  : ${NAMESPACE:=rethinkdb}
  # try to pick up first different ip from endpoints
  MYHOST=$(ip addr | grep 'state UP' -A2 | tail -n1 | awk '{print $2}' | cut -f1  -d'/')
  URL="${KUBERNETES_RO_SERVICE_HOST}/api/v1beta1/endpoints/rethinkdb-driver?namespace=${NAMESPACE}"
  IP=$(curl -s ${URL} | jq -s -r --arg h "${MYHOST}" '.[0].endpoints | [ .[]? | split(":") ] | map(select(.[0] != $h)) | .[0][0]') || exit 1
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
