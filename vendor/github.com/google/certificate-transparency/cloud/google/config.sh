DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" == "" ]; then
  echo "Usage $0: <config-file>"
  exit 1
fi
source ${DIR}/util.sh
source $1

export ETCD_NUM_REPLICAS_PER_ZONE=1
export ETCD_DISK_SIZE=200GB
export ETCD_BASE_NAME="${CLUSTER}-etcd"
export ETCD_MACHINE_TYPE=n1-standard-2
declare -a ETCD_ZONES ETCD_MACHINES ETCD_DISKS
export ETCD_ZONES ETCD_MACHINES ETCD_DISKS
export ETCD_NUM_REPLICAS=0
for z in ${ZONES}; do
  for i in $(seq ${ETCD_NUM_REPLICAS_PER_ZONE}); do
    ETCD_ZONES[${ETCD_NUM_REPLICAS}]="${REGION}-${z}"
    ETCD_MACHINES[${ETCD_NUM_REPLICAS}]="${CLUSTER}-etcd-${z}-${i}"
    ETCD_DISKS[${ETCD_NUM_REPLICAS}]="${CLUSTER}-etcd-disk-${z}-${i}"
    ETCD_NUM_REPLICAS=$((${ETCD_NUM_REPLICAS} + 1))
  done
done
export ETCD_SERVER_LIST=`AppendAndJoin ":4001" "," ${ETCD_MACHINES[@]}`

if [ "${INSTANCE_TYPE}" == "mirror" ]; then
  if [ "${MIRROR_TARGET_URL}" == "" ]; then
    echo "Must set MIRROR_TARGET_URL for mirror instance type."
    exit 1
  fi
  if [ "${MIRROR_TARGET_PUBLIC_KEY}" == "" ]; then
    echo "Must set MIRROR_TARGET_PUBLIC_KEY for mirror instance type."
    exit 1
  fi

  export MIRROR_NUM_REPLICAS_PER_ZONE=${MIRROR_NUM_REPLICAS_PER_ZONE:-2}
  export MIRROR_DISK_SIZE=${MIRROR_DISK_SIZE:-200GB}
  export MIRROR_MACHINE_TYPE=${MIRROR_MACHINE_TYPE:-n1-highmem-2}
  export MIRROR_BASE_NAME="${CLUSTER}-mirror"
  declare -a MIRROR_ZONES MIRROR_MACHINES MIRROR_DISKS MIRROR_META
  export MIRROR_ZONES MIRROR_MACHINES MIRROR_DISKS
  export MIRROR_NUM_REPLICAS=0
  for z in ${ZONES}; do
    for i in $(seq ${MIRROR_NUM_REPLICAS_PER_ZONE}); do
      MIRROR_ZONES[${MIRROR_NUM_REPLICAS}]="${REGION}-${z}"
      MIRROR_MACHINES[${MIRROR_NUM_REPLICAS}]="${CLUSTER}-mirror-${z}-${i}"
      MIRROR_DISKS[${MIRROR_NUM_REPLICAS}]="${CLUSTER}-mirror-disk-${z}-${i}"
      MIRROR_META[${MIRROR_NUM_REPLICAS}]=$(
          sed --e "s^@@PROJECT@@^${PROJECT}^
                   s^@@ETCD_SERVERS@@^${ETCD_SERVER_LIST}^
                   s^@@CONTAINER_HOST@@^${MIRROR_MACHINES[${MIRROR_NUM_REPLICAS}]}^
                   s^@@TARGET_LOG_URL@@^${MIRROR_TARGET_URL}^
                   s^@@TARGET_LOG_TLS_VERSION@@^${MIRROR_TARGET_TLS_VERSION:-tlsv12}^
                   s^@@TARGET_LOG_PUBLIC_KEY@@^${MIRROR_TARGET_PUBLIC_KEY}^
                   s^@@MONITORING@@^${MONITORING}^" \
                      < ${DIR}/mirror_container.yaml)
      MIRROR_NUM_REPLICAS=$((${MIRROR_NUM_REPLICAS} + 1))
    done
  done
elif [ "${INSTANCE_TYPE}" == "log" ]; then
  export LOG_NUM_REPLICAS_PER_ZONE=${LOG_NUM_REPLICAS_PER_ZONE:-2}
  export LOG_DISK_SIZE=${LOG_DISK_SIZE:-200GB}
  export LOG_MACHINE_TYPE=${LOG_MACHINE_TYPE:-n1-highmem-2}
  export LOG_BASE_NAME="${CLUSTER}-log"
  declare -a LOG_ZONES LOG_MACHINES LOG_DISKS
  export LOG_ZONES LOG_MACHINES LOG_DISKS
  export LOG_NUM_REPLICAS=0
  for z in ${ZONES}; do
    for i in $(seq ${LOG_NUM_REPLICAS_PER_ZONE}); do
      LOG_ZONES[${LOG_NUM_REPLICAS}]="${REGION}-${z}"
      LOG_MACHINES[${LOG_NUM_REPLICAS}]="${CLUSTER}-log-${z}-${i}"
      LOG_DISKS[${LOG_NUM_REPLICAS}]="${CLUSTER}-log-disk-${z}-${i}"
      LOG_META[${LOG_NUM_REPLICAS}]=$(
          sed --e "s^@@PROJECT@@^${PROJECT}^
                   s^@@ETCD_SERVERS@@^${ETCD_SERVER_LIST}^
                   s^@@CONTAINER_HOST@@^${LOG_MACHINES[${LOG_NUM_REPLICAS}]}^
                   s^@@MONITORING@@^${MONITORING}^" \
                      < ${DIR}/log_container.yaml)
      LOG_NUM_REPLICAS=$((${LOG_NUM_REPLICAS} + 1))
    done
  done
else
  echo "INSTANCE_TYPE must be set to either 'mirror' or 'log'"
  exit 1
fi


if [ "${MONITORING}" == "prometheus" ]; then
  export PROMETHEUS_NUM_REPLICAS_PER_ZONE=1
  export PROMETHEUS_DISK_SIZE=50GB
  export PROMETHEUS_BASE_NAME="${CLUSTER}-prometheus"
  export PROMETHEUS_MACHINE_TYPE=n1-standard-1
  declare -a PROMETHEUS_ZONES PROMETHEUS_MACHINES MIRROR_DISKS
  export PROMETHEUS_ZONES PROMETHEUS_MACHINES MIRROR_DISKS
  export PROMETHEUS_NUM_REPLICAS=0
  for z in ${ZONES}; do
    for i in $(seq ${PROMETHEUS_NUM_REPLICAS_PER_ZONE}); do
      PROMETHEUS_ZONES[${PROMETHEUS_NUM_REPLICAS}]="${REGION}-${z}"
      PROMETHEUS_MACHINES[${PROMETHEUS_NUM_REPLICAS}]="${CLUSTER}-prometheus-${z}-${i}"
      PROMETHEUS_DISKS[${PROMETHEUS_NUM_REPLICAS}]="${CLUSTER}-prometheus-disk-${z}-${i}"
      PROMETHEUS_NUM_REPLICAS=$((${PROMETHEUS_NUM_REPLICAS} + 1))
    done
  done
elif [ "${MONITORING}" != "gcm" ]; then
  echo "MONITORING must be set to either 'prometheus' or 'gcm'"
  exit 1
fi


echo "============================================================="
echo "Cluster config:"
echo "PROJECT:     ${PROJECT}"
echo "TYPE:        ${INSTANCE_TYPE}"
echo "CLUSTER:     ${CLUSTER}"
echo "REGION:      ${REGION}"
echo "ZONES:       ${ZONES}"
echo "Num etcd:    ${ETCD_NUM_REPLICAS}"
echo "Num mirrors: ${MIRROR_NUM_REPLICAS}"
echo "Monitoring:  ${MONITORING}"
if [ "${MONITORING}" == "prometheus" ]; then
  echo "Num prom:    ${PROMETHEUS_NUM_REPLICAS}"
fi
echo "============================================================="
echo
