#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# A library of helper functions that each provider hosting Kubernetes must implement to use cluster/kube-*.sh scripts.

source $(dirname ${BASH_SOURCE})/${KUBE_CONFIG_FILE-"config-default.sh"}

function detect-master () {
	echo "KUBE_MASTER_IP: $KUBE_MASTER_IP"
	echo "KUBE_MASTER: $KUBE_MASTER"
}

# Get minion IP addresses and store in KUBE_MINION_IP_ADDRESSES[]
function detect-minions {
	echo "Minions already detected"
}

# Verify prereqs on host machine
function verify-prereqs {
  for x in vagrant virtualbox; do
    if [ "$(which $x)" == "" ]; then
      echo "Can't find $x in PATH, please fix and retry."
      exit 1
    fi
  done
}

# Instantiate a kubernetes cluster
function kube-up {
	vagrant up
}

# Delete a kubernetes cluster
function kube-down {
	vagrant destroy -f
}

# Update a kubernetes cluster with latest source
function kube-push {
	vagrant provision
}

# Execute prior to running tests to build a release if required for env
function test-build-release {
	echo "Vagrant provider can skip release build"
}

# Execute prior to running tests to initialize required structure
function test-setup {
	echo "Vagrant test setup complete"
}

# Execute after running tests to perform any required clean-up
function test-teardown {
	echo "Vagrant ignores tear-down"
}

# Set the {user} and {password} environment values required to interact with provider
function get-password {
	export user=vagrant
	export passwd=vagrant
	echo "Using credentials: $user:$passwd"
}

