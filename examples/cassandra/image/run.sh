#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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
CFG=/etc/cassandra/cassandra.yaml
CASSANDRA_RPC_ADDRESS="${CASSANDRA_RPC_ADDRESS:-0.0.0.0}"
CASSANDRA_NUM_TOKENS="${CASSANDRA_NUM_TOKENS:-32}"
CASSANDRA_CLUSTER_NAME="${CASSANDRA_CLUSTER_NAME:='Test Cluster'}"
CASSANDRA_LISTEN_ADDRESS=${POD_IP}
CASSANDRA_BROADCAST_ADDRESS=${POD_IP}
CASSANDRA_BROADCAST_RPC_ADDRESS=${POD_IP}

# TODO what else needs to be modified

for yaml in \
  broadcast_address \
	broadcast_rpc_address \
	cluster_name \
	listen_address \
	num_tokens \
	rpc_address \
; do
  var="CASSANDRA_${yaml^^}"
	val="${!var}"
	if [ "$val" ]; then
		sed -ri 's/^(# )?('"$yaml"':).*/\2 '"$val"'/' "$CFG"
	fi
done

# Eventual do snitch $DC && $RACK?
#if [[ $SNITCH ]]; then
#  sed -i -e "s/endpoint_snitch: SimpleSnitch/endpoint_snitch: $SNITCH/" $CONFIG/cassandra.yaml
#fi
#if [[ $DC && $RACK ]]; then
#  echo "dc=$DC" > $CONFIG/cassandra-rackdc.properties
#  echo "rack=$RACK" >> $CONFIG/cassandra-rackdc.properties
#fi

#
# see if this is needed
#echo "JVM_OPTS=\"\$JVM_OPTS -Djava.rmi.server.hostname=$IP\"" >> $CASSANDRA_CONFIG/cassandra-env.sh
#

# FIXME create README for these args
echo "Starting Cassandra on $POD_IP"
echo CASSANDRA_RPC_ADDRESS ${CASSANDRA_RPC_ADDRESS}
echo CASSANDRA_NUM_TOKENS ${CASSANDRA_NUM_TOKENS}
echo CASSANDRA_CLUSTER_NAME ${CASSANDRA_CLUSTER_NAME}
echo CASSANDRA_LISTEN_ADDRESS ${POD_IP}
echo CASSANDRA_BROADCAST_ADDRESS ${POD_IP}
echo CASSANDRA_BROADCAST_RPC_ADDRESS ${POD_IP}

export CLASSPATH=/kubernetes-cassandra.jar
cassandra -f
