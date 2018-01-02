#!/bin/bash
set -e
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
GCLOUD="gcloud"
if [ "$1" == "" ]; then
  echo "Usage: $0 <config-file>"
  exit 1
fi
CONFIG_FILE="$1"

. ${DIR}/config.sh ${CONFIG_FILE}

case "${INSTANCE_TYPE}" in
  "mirror")
    eval $(typeset -A -p MIRROR_ZONES | sed 's/MIRROR_ZONES=/NODE_ZONES=/')
    eval $(typeset -A -p MIRROR_MACHINES | \
           sed 's/MIRROR_MACHINES=/NODE_MACHINES=/')
    export NODE_NUM_REPLICAS=${MIRROR_NUM_REPLICAS}
    export NODE_ZONES
    ;;
  "log")
    eval $(typeset -A -p LOG_ZONES | sed 's/LOG_ZONES=/NODE_ZONES=/')
    eval $(typeset -A -p LOG_MACHINES | \
           sed 's/LOG_MACHINES=/NODE_MACHINES=/')
    export NODE_NUM_REPLICAS=${LOG_NUM_REPLICAS}
    export NODE_ZONES
   ;;
  *)
    echo "Unknown INSTANCE_TYPE: ${INSTANCE_TYPE}"
    exit 1
esac

ZONE_LIST=$(echo ${NODE_ZONES[*]} | tr " " "\n" | sort | uniq )

echo "============================================================="
echo "Creating network rules..."
gcloud compute http-health-checks create get-sth-check \
    --port 80 \
    --request-path /ct/v1/get-sth
gcloud compute firewall-rules create ${INSTANCE_TYPE}-node-80 \
    --allow tcp:80 \
    --target-tags ${INSTANCE_TYPE}-node

for zone in ${ZONE_LIST}; do
  gcloud compute instance-groups unmanaged \
      create "${INSTANCE_TYPE}-group-${zone}" \
      --zone ${zone} &
done
wait

for i in `seq 0 $((${NODE_NUM_REPLICAS} - 1))`; do
  gcloud compute instance-groups unmanaged add-instances \
      "${INSTANCE_TYPE}-group-${NODE_ZONES[${i}]}" \
      --zone ${NODE_ZONES[${i}]} \
      --instances ${NODE_MACHINES[${i}]} &
done
wait

gcloud compute addresses create "${INSTANCE_TYPE}-ip" \
    --global
export EXTERNAL_IP=$(gcloud compute addresses list "${INSTANCE_TYPE}-ip" |
                     awk -- "/${INSTANCE_TYPE}-ip/ {print \$2}")
echo "Service IP: ${EXTERNAL_IP}"


gcloud compute backend-services create "${INSTANCE_TYPE}-lb-backend" \
    --http-health-check "get-sth-check" \
    --timeout "30"

for zone in ${ZONE_LIST}; do
  gcloud compute backend-services add-backend "${INSTANCE_TYPE}-lb-backend" \
    --instance-group "${INSTANCE_TYPE}-group-${zone}" \
    --zone ${zone} \
    --balancing-mode "UTILIZATION" \
    --capacity-scaler "1" \
    --max-utilization "0.8"
done


gcloud compute url-maps create "${INSTANCE_TYPE}-lb-url-map" \
    --default-service "${INSTANCE_TYPE}-lb-backend"

gcloud compute target-http-proxies create "${INSTANCE_TYPE}-lb-http-proxy" \
    --url-map "${INSTANCE_TYPE}-lb-url-map"

gcloud compute forwarding-rules create "${INSTANCE_TYPE}-fwd" \
    --global \
    --address "${EXTERNAL_IP}" \
    --ip-protocol "TCP" \
    --port-range "80" \
    --target-http-proxy "${INSTANCE_TYPE}-lb-http-proxy"

echo "============================================================="
echo "External IPs:"
gcloud compute forwarding-rules list
echo "============================================================="

