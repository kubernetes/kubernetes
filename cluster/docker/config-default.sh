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

## Contains configuration values for the bare-metal cluster

# (0) Versions
export ETCD_VERSION=${ETCD_VERSION:-"2.0.12"}
export K8S_VERSION=${K8S_VERSION:-"1.0.3"}

# (1) Cluster topology
# And separated with blank space:
# e.g.
#
# export NODES="user@ip_1 user@ip_2 user@ip_3" 
# export MASTER="user@ip_1"

# Must be set
export NODES=${NODES:-}
# Must be set
export MASTER=${MASTER:-}
# Modify this value may fail validate-cluster if MASTER_IP is not public accessible
export MASTER_IP=${MASTER#*@}
# Set to 'yes' if you only want to add node to a existing cluster, then we'll not deploy master
export NODE_ONLY=${NODE_ONLY:-"no"}
# Set to 'yes' if you want to customize your k8s master
export MASTER_CONF=${MASTER_CONF:-"no"}
# Set to your own infra provider like 'gce', 'aws' etc.
export INFRA=${INFRA:-"baremetal"}

# (2) SSH OPTS
# define the SSH OPTS.
export SSH_OPTS=${SSH_OPTS:-"-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR"}

# (3) Networking
export FLANNEL_VERSION=${FLANNEL_VERSION:-"0.5.3"}
# Define the IP range used for flannel overlay network, should not conflict with SERVICE_CLUSTER_IP_RANGE
export FLANNEL_NET=${FLANNEL_NET:-10.16.0.0/16}

