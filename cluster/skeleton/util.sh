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

# This script contains the helper functions that each provider hosting
# Kubernetes must implement to use cluster/kube-*.sh scripts.

# Must ensure that the following ENV vars are set
function detect-master {
	echo "KUBE_MASTER_IP: $KUBE_MASTER_IP" 1>&2
	echo "KUBE_MASTER: $KUBE_MASTER" 1>&2
}

# Get node names if they are not static.
function detect-node-names {
	echo "NODE_NAMES: [${NODE_NAMES[*]}]" 1>&2
}

# Get node IP addresses and store in KUBE_NODE_IP_ADDRESSES[]
function detect-nodes {
	echo "KUBE_NODE_IP_ADDRESSES: [${KUBE_NODE_IP_ADDRESSES[*]}]" 1>&2
}

# Verify prereqs on host machine
function verify-prereqs {
	echo "Skeleton Provider: verify-prereqs not implemented" 1>&2
}

# Validate a kubernetes cluster
function validate-cluster {
	# by default call the generic validate-cluster.sh script, customizable by
	# any cluster provider if this does not fit.
	"${KUBE_ROOT}/cluster/validate-cluster.sh"
}

# Instantiate a kubernetes cluster
function kube-up {
	echo "Skeleton Provider: kube-up not implemented" 1>&2
}

# Delete a kubernetes cluster
function kube-down {
	echo "Skeleton Provider: kube-down not implemented" 1>&2
}

# Update a kubernetes cluster
function kube-push {
	echo "Skeleton Provider: kube-push not implemented" 1>&2
}

# Prepare update a kubernetes component
function prepare-push {
	echo "Skeleton Provider: prepare-push not implemented" 1>&2
}

# Update a kubernetes master
function push-master {
	echo "Skeleton Provider: push-master not implemented" 1>&2
}

# Update a kubernetes node
function push-node {
	echo "Skeleton Provider: push-node not implemented" 1>&2
}

# Execute prior to running tests to build a release if required for env
function test-build-release {
	echo "Skeleton Provider: test-build-release not implemented" 1>&2
}

# Execute prior to running tests to initialize required structure
function test-setup {
	echo "Skeleton Provider: test-setup not implemented" 1>&2
}

# Execute after running tests to perform any required clean-up
function test-teardown {
	echo "Skeleton Provider: test-teardown not implemented" 1>&2
}

function prepare-e2e {
	echo "Skeleton Provider: prepare-e2e not implemented" 1>&2
}
