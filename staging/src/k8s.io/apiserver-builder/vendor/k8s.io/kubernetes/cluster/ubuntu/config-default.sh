#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

## Contains configuration values for the Ubuntu cluster

# Define all your cluster nodes, MASTER node comes first"
# And separated with blank space like <user_1@ip_1> <user_2@ip_2> <user_3@ip_3>
export nodes=${nodes:-"vcap@10.10.103.250 vcap@10.10.103.162 vcap@10.10.103.223"}

# Define all your nodes role: a(master) or i(minion) or ai(both master and minion),
# Roles must be the same order with the nodes.
roles=${roles:-"ai i i"}
# If it practically impossible to set an array as an environment variable
# from a script, so assume variable is a string then convert it to an array
export roles_array=($roles)

# Define minion numbers
export NUM_NODES=${NUM_NODES:-3}
# define the IP range used for service cluster IPs.
# according to rfc 1918 ref: https://tools.ietf.org/html/rfc1918 choose a private ip range here.
export SERVICE_CLUSTER_IP_RANGE=${SERVICE_CLUSTER_IP_RANGE:-192.168.3.0/24}  # formerly PORTAL_NET
# define the IP range used for flannel overlay network, should not conflict with above SERVICE_CLUSTER_IP_RANGE

# The Ubuntu scripting supports two ways of networking: Flannel and
# CNI.  To use CNI: (1) put a CNI configuration file, whose basename
# is the configured network type plus ".conf", somewhere on the driver
# machine (the one running `kube-up.sh`) and set CNI_PLUGIN_CONF to a
# pathname of that file, (2) put one or more executable binaries on
# the driver machine and set CNI_PLUGIN_EXES to a space-separated list
# of their pathnames, and (3) set CNI_KUBELET_TRIGGER to identify an
# appropriate service on which to trigger the start and stop of the
# kubelet on non-master machines.  For (1) and (2) the pathnames may
# be relative, in which case they are relative to kubernetes/cluster.
# If either of CNI_PLUGIN_CONF or CNI_PLUGIN_EXES is undefined or has
# a zero length value then Flannel will be used instead of CNI.

export CNI_PLUGIN_CONF CNI_PLUGIN_EXES CNI_KUBELET_TRIGGER
CNI_PLUGIN_CONF=${CNI_PLUGIN_CONF:-""}
CNI_PLUGIN_EXES=${CNI_PLUGIN_EXES:-""}
CNI_KUBELET_TRIGGER=${CNI_KUBELET_TRIGGER:-networking}

# Flannel networking is used if CNI networking is not.  The following
# variable defines the CIDR block from which cluster addresses are
# drawn.
export FLANNEL_NET=${FLANNEL_NET:-172.16.0.0/16}

# If Flannel networking is used then the following variable can be
# used to customize the Flannel backend.  The variable's value should
# be a JSON object.  An empty string means to use the default, which
# is `{"Type": "vxlan"}`.  See
# https://github.com/coreos/flannel#configuration for details on
# configuring Flannel.

export FLANNEL_BACKEND
FLANNEL_BACKEND=''

# Optionally add other contents to the Flannel configuration JSON
# object normally stored in etcd as /coreos.com/network/config.  Use
# JSON syntax suitable for insertion into a JSON object constructor
# after other field name:value pairs.  For example:
# FLANNEL_OTHER_NET_CONFIG=', "SubnetMin": "172.16.10.0", "SubnetMax": "172.16.90.0"'

export FLANNEL_OTHER_NET_CONFIG
FLANNEL_OTHER_NET_CONFIG=${FLANNEL_OTHER_NET_CONFIG:-""}

# Admission Controllers to invoke prior to persisting objects in
# cluster.  If we included ResourceQuota, we should keep it at the end
# of the list to prevent incrementing quota usage prematurely.  The
# list below is what
# http://kubernetes.io/docs/admin/admission-controllers/ recommends
# for release >= 1.4.0; see that doc for the recommended settings for
# earlier releases.

export ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,ResourceQuota

# Path to the pod manifest file or directory of files of kubelet
export KUBELET_POD_MANIFEST_PATH=${KUBELET_POD_MANIFEST_PATH:-""}

# A port range to reserve for services with NodePort visibility
SERVICE_NODE_PORT_RANGE=${SERVICE_NODE_PORT_RANGE:-"30000-32767"}

# Optional: Enable node logging.
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=${LOGGING_DESTINATION:-elasticsearch}

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING=false
ELASTICSEARCH_LOGGING_REPLICAS=${ELASTICSEARCH_LOGGING_REPLICAS:-1}

# Optional: When set to true, heapster, Influxdb and Grafana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-true}"

# Extra options to set on the Docker command line.  This is useful for setting
# --insecure-registry for local registries.
DOCKER_OPTS=${DOCKER_OPTS:-""}

# Extra options to set on the kube-proxy command line.  This is useful
# for selecting the iptables proxy-mode, for example.
KUBE_PROXY_EXTRA_OPTS=${KUBE_PROXY_EXTRA_OPTS:-""}

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
# DNS_SERVER_IP must be a IP in SERVICE_CLUSTER_IP_RANGE
DNS_SERVER_IP=${DNS_SERVER_IP:-"192.168.3.10"}
DNS_DOMAIN=${DNS_DOMAIN:-"cluster.local"}

# Optional: Enable DNS horizontal autoscaler
ENABLE_DNS_HORIZONTAL_AUTOSCALER="${KUBE_ENABLE_DNS_HORIZONTAL_AUTOSCALER:-false}"

# Optional: Install Kubernetes UI
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"

# Optional: Enable setting flags for kube-apiserver to turn on behavior in active-dev
#RUNTIME_CONFIG=""

# Optional: Add http or https proxy when download easy-rsa.
# Add environment variable separated with blank space like "http_proxy=http://10.x.x.x:8080 https_proxy=https://10.x.x.x:8443"
PROXY_SETTING=${PROXY_SETTING:-""}

# Optional: Allows kubelet/kube-api to be run in privileged mode
ALLOW_PRIVILEGED=${ALLOW_PRIVILEGED:-"false"}

DEBUG=${DEBUG:-"false"}

# Add SSH_OPTS: Add this to config ssh port
SSH_OPTS="-oPort=22 -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR"
