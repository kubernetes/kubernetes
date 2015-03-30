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

# A library of helper functions that each provider hosting LMKTFY must implement to use cluster/lmktfy-*.sh scripts.

# Must ensure that the following ENV vars are set
function detect-master {
	echo "LMKTFY_MASTER_IP: $LMKTFY_MASTER_IP"
	echo "LMKTFY_MASTER: $LMKTFY_MASTER"
}

# Get minion names if they are not static.
function detect-minion-names {
        echo "MINION_NAMES: ${MINION_NAMES[*]}"
}

# Get minion IP addresses and store in LMKTFY_MINION_IP_ADDRESSES[]
function detect-minions {
	echo "LMKTFY_MINION_IP_ADDRESSES=[]"
}

# Verify prereqs on host machine
function verify-prereqs {
	echo "TODO"
}

# Instantiate a lmktfy cluster
function lmktfy-up {
	echo "TODO"
}

# Delete a lmktfy cluster
function lmktfy-down {
	echo "TODO"
}

# Update a lmktfy cluster with latest source
function lmktfy-push {
	echo "TODO"
}

# Execute prior to running tests to build a release if required for env
function test-build-release {
	echo "TODO"
}

# Execute prior to running tests to initialize required structure
function test-setup {
	echo "TODO"
}

# Execute after running tests to perform any required clean-up
function test-teardown {
	echo "TODO"
}

# Set the {LMKTFY_USER} and {LMKTFY_PASSWORD} environment values required to interact with provider
function get-password {
	echo "TODO"
}
