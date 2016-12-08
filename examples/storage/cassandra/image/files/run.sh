#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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
CASSANDRA_CONF_DIR=/etc/cassandra
CASSANDRA_CFG=$CASSANDRA_CONF_DIR/cassandra.yaml

# we are doing StatefulSet or just setting our seeds
if [ -z "$CASSANDRA_SEEDS" ]; then
  HOSTNAME=$(hostname -f)
fi

# The following vars relate to there counter parts in $CASSANDRA_CFG
# for instance rpc_address
CASSANDRA_RPC_ADDRESS="${CASSANDRA_RPC_ADDRESS:-0.0.0.0}"
CASSANDRA_NUM_TOKENS="${CASSANDRA_NUM_TOKENS:-32}"
CASSANDRA_CLUSTER_NAME="${CASSANDRA_CLUSTER_NAME:='Test Cluster'}"
CASSANDRA_LISTEN_ADDRESS=${POD_IP:-$HOSTNAME}
CASSANDRA_BROADCAST_ADDRESS=${POD_IP:-$HOSTNAME}
CASSANDRA_BROADCAST_RPC_ADDRESS=${POD_IP:-$HOSTNAME}
CASSANDRA_DISK_OPTIMIZATION_STRATEGY="${CASSANDRA_DISK_OPTIMIZATION_STRATEGY:-ssd}"
CASSANDRA_MIGRATION_WAIT="${CASSANDRA_MIGRATION_WAIT:-1}"
CASSANDRA_ENDPOINT_SNITCH="${CASSANDRA_ENDPOINT_SNITCH:-SimpleSnitch}"
CASSANDRA_DC="${CASSANDRA_DC}"
CASSANDRA_RACK="${CASSANDRA_RACK}"
CASSANDRA_RING_DELAY="${CASSANDRA_RING_DELAY:-30000}"
CASSANDRA_AUTO_BOOTSTRAP="${CASSANDRA_AUTO_BOOTSTRAP:-true}"
CASSANDRA_SEEDS="${CASSANDRA_SEEDS:false}"
CASSANDRA_SEED_PROVIDER="${CASSANDRA_SEED_PROVIDER:-org.apache.cassandra.locator.SimpleSeedProvider}"
CASSANDRA_AUTO_BOOTSTRAP="${CASSANDRA_AUTO_BOOTSTRAP:false}"

# Turn off JMX auth
CASSANDRA_OPEN_JMX="${CASSANDRA_OPEN_JMX:-false}"
# send GC to STDOUT
CASSANDRA_GC_STDOUT="${CASSANDRA_GC_STDOUT:-false}"

echo Starting Cassandra on ${CASSANDRA_LISTEN_ADDRESS}
echo CASSANDRA_CONF_DIR ${CASSANDRA_CONF_DIR}
echo CASSANDRA_CFG ${CASSANDRA_CFG}
echo CASSANDRA_AUTO_BOOTSTRAP ${CASSANDRA_AUTO_BOOTSTRAP}
echo CASSANDRA_BROADCAST_ADDRESS ${CASSANDRA_BROADCAST_ADDRESS}
echo CASSANDRA_BROADCAST_RPC_ADDRESS ${CASSANDRA_BROADCAST_RPC_ADDRESS}
echo CASSANDRA_CLUSTER_NAME ${CASSANDRA_CLUSTER_NAME}
echo CASSANDRA_COMPACTION_THROUGHPUT_MB_PER_SEC ${CASSANDRA_COMPACTION_THROUGHPUT_MB_PER_SEC}
echo CASSANDRA_CONCURRENT_COMPACTORS ${CASSANDRA_CONCURRENT_COMPACTORS}
echo CASSANDRA_CONCURRENT_READS ${CASSANDRA_CONCURRENT_READS}
echo CASSANDRA_CONCURRENT_WRITES ${CASSANDRA_CONCURRENT_WRITES}
echo CASSANDRA_COUNTER_CACHE_SIZE_IN_MB ${CASSANDRA_COUNTER_CACHE_SIZE_IN_MB}
echo CASSANDRA_DC ${CASSANDRA_DC}
echo CASSANDRA_DISK_OPTIMIZATION_STRATEGY ${CASSANDRA_DISK_OPTIMIZATION_STRATEGY}
echo CASSANDRA_ENDPOINT_SNITCH ${CASSANDRA_ENDPOINT_SNITCH}
echo CASSANDRA_GC_WARN_THRESHOLD_IN_MS ${CASSANDRA_GC_WARN_THRESHOLD_IN_MS}
echo CASSANDRA_INTERNODE_COMPRESSION ${CASSANDRA_INTERNODE_COMPRESSION}
echo CASSANDRA_KEY_CACHE_SIZE_IN_MB ${CASSANDRA_KEY_CACHE_SIZE_IN_MB}
echo CASSANDRA_LISTEN_ADDRESS ${CASSANDRA_LISTEN_ADDRESS}
echo CASSANDRA_LISTEN_INTERFACE ${CASSANDRA_LISTEN_INTERFACE}
echo CASSANDRA_MEMTABLE_ALLOCATION_TYPE ${CASSANDRA_MEMTABLE_ALLOCATION_TYPE}
echo CASSANDRA_MEMTABLE_CLEANUP_THRESHOLD ${CASSANDRA_MEMTABLE_CLEANUP_THRESHOLD}
echo CASSANDRA_MEMTABLE_FLUSH_WRITERS ${CASSANDRA_MEMTABLE_FLUSH_WRITERS}
echo CASSANDRA_MIGRATION_WAIT ${CASSANDRA_MIGRATION_WAIT}
echo CASSANDRA_NUM_TOKENS ${CASSANDRA_NUM_TOKENS}
echo CASSANDRA_RACK ${CASSANDRA_RACK}
echo CASSANDRA_RING_DELAY ${CASSANDRA_RING_DELAY}
echo CASSANDRA_RPC_ADDRESS ${CASSANDRA_RPC_ADDRESS}
echo CASSANDRA_RPC_INTERFACE ${CASSANDRA_RPC_INTERFACE}
echo CASSANDRA_SEEDS ${CASSANDRA_SEEDS}
echo CASSANDRA_SEED_PROVIDER ${CASSANDRA_SEED_PROVIDER}


# if DC and RACK are set, use GossipingPropertyFileSnitch
if [[ $CASSANDRA_DC && $CASSANDRA_RACK ]]; then
  echo "dc=$CASSANDRA_DC" > $CASSANDRA_CONF_DIR/cassandra-rackdc.properties
  echo "rack=$CASSANDRA_RACK" >> $CASSANDRA_CONF_DIR/cassandra-rackdc.properties
  CASSANDRA_ENDPOINT_SNITCH="GossipingPropertyFileSnitch"
fi

if [ -n "$CASSANDRA_MAX_HEAP" ]; then
  sed -ri "s/^(#)?-Xmx[0-9]+.*/-Xmx$CASSANDRA_MAX_HEAP/" "$CASSANDRA_CONF_DIR/jvm.options"
  sed -ri "s/^(#)?-Xms[0-9]+.*/-Xms$CASSANDRA_MAX_HEAP/" "$CASSANDRA_CONF_DIR/jvm.options"
fi

if [ -n "$CASSANDRA_REPLACE_NODE" ]; then
   echo "-Dcassandra.replace_address=$CASSANDRA_REPLACE_NODE/" >> "$CASSANDRA_CONF_DIR/jvm.options"
fi

for rackdc in dc rack; do
  var="CASSANDRA_${rackdc^^}"
  val="${!var}"
  if [ "$val" ]; then
	sed -ri 's/^('"$rackdc"'=).*/\1 '"$val"'/' "$CASSANDRA_CONF_DIR/cassandra-rackdc.properties"
  fi
done

# TODO what else needs to be modified
for yaml in \
  broadcast_address \
  broadcast_rpc_address \
  cluster_name \
  disk_optimization_strategy \
  endpoint_snitch \
  listen_address \
  num_tokens \
  rpc_address \
  start_rpc \
  key_cache_size_in_mb \
  concurrent_reads \
  concurrent_writes \
  memtable_cleanup_threshold \
  memtable_allocation_type \
  memtable_flush_writers \
  concurrent_compactors \
  compaction_throughput_mb_per_sec \
  counter_cache_size_in_mb \
  internode_compression \
  endpoint_snitch \
  gc_warn_threshold_in_ms \
  listen_interface \
  rpc_interface \
  ; do
  var="CASSANDRA_${yaml^^}"
  val="${!var}"
  if [ "$val" ]; then
    sed -ri 's/^(# )?('"$yaml"':).*/\2 '"$val"'/' "$CASSANDRA_CFG"
  fi
done

echo "auto_bootstrap: ${CASSANDRA_AUTO_BOOTSTRAP}" >> $CASSANDRA_CFG

# set the seed to itself.  This is only for the first pod, otherwise
# it will be able to get seeds from the seed provider
if [[ $CASSANDRA_SEEDS == 'false' ]]; then
  sed -ri 's/- seeds:.*/- seeds: "'"$POD_IP"'"/' $CASSANDRA_CFG
else # if we have seeds set them.  Probably StatefulSet
  sed -ri 's/- seeds:.*/- seeds: "'"$CASSANDRA_SEEDS"'"/' $CASSANDRA_CFG
fi

sed -ri 's/- class_name: SEED_PROVIDER/- class_name: '"$CASSANDRA_SEED_PROVIDER"'/' $CASSANDRA_CFG

# send gc to stdout
if [[ $CASSANDRA_GC_STDOUT == 'true' ]]; then
  sed -ri 's/ -Xloggc:\/var\/log\/cassandra\/gc\.log//' $CASSANDRA_CONF_DIR/cassandra-env.sh
fi

# enable RMI and JMX to work on one port
echo "JVM_OPTS=\"\$JVM_OPTS -Djava.rmi.server.hostname=$POD_IP\"" >> $CASSANDRA_CONF_DIR/cassandra-env.sh

# getting WARNING messages with Migration Service
echo "-Dcassandra.migration_task_wait_in_seconds=${CASSANDRA_MIGRATION_WAIT}" >> $CASSANDRA_CONF_DIR/jvm.options
echo "-Dcassandra.ring_delay_ms=${CASSANDRA_RING_DELAY}" >> $CASSANDRA_CONF_DIR/jvm.options

if [[ $CASSANDRA_OPEN_JMX == 'true' ]]; then
  export LOCAL_JMX=no
  sed -ri 's/ -Dcom\.sun\.management\.jmxremote\.authenticate=true/ -Dcom\.sun\.management\.jmxremote\.authenticate=false/' $CASSANDRA_CONF_DIR/cassandra-env.sh
  sed -ri 's/ -Dcom\.sun\.management\.jmxremote\.password\.file=\/etc\/cassandra\/jmxremote\.password//' $CASSANDRA_CONF_DIR/cassandra-env.sh
fi

export CLASSPATH=/kubernetes-cassandra.jar
cassandra -R -f
