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
# Kubermark must implement to use test/kubemark/start-kubemark.sh and
# test/kubemark/stop-kubemark.sh scripts.


# This function should create a machine instance for the master along
# with any/all of the following resources:
# - Attach a PD to the master (optionally 1 more for storing events)
# - A public IP address for the master ($MASTER_IP)
# - A network firewall rule allowing all TCP traffic on port 443 in master
#   Note: This step is compulsory in order for kubemark to work
#
# ENV vars that should be defined by the end of this function:
# 1. MASTER_IP
# 2. MASTER_NAME
function create-master-instance-with-resources {
	echo "MASTER_IP: $MASTER_IP" 1>&2
	echo "MASTER_NAME: $MASTER_NAME" 1>&2
}

# This function should wait until the master is reachable through SSH.
function wait-for-master-ssh-reachability {
	echo "Waiting for master to be SSH reachable..." 1>&2
}

# This function should execute the command('$1') on the master machine
# (possibly through SSH).
function execute-cmd-on-master() {
	echo "Executing command on the master" 1>&2
}

# This function should act as an scp for the kubemark cluster, which copies
# the files given by the first n-1 arguments to the remote location given
# by the n^th argument.
function copy-files() {
	echo "Copying files" 1>&2
}

# This function should delete the master instance along with all the
# resources that have been allocated inside the function
# 'create-master-instance-with-resources' above.
function delete-master-instance-and-resources {
	echo "Deleting master instance and its allocated resources" 1>&2
}
