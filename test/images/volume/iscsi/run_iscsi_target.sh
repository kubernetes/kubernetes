#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

# This script does not run any daemon, it only configures iSCSI target (=server)
# in kernel. It is possible to run this script multiple times on a single
# node, each run will create its own IQN and LUN.

# Kubernetes must provide unique name.
IQN=$1

function start()
{
    # Create new IQN (iSCSI Qualified Name)
    targetcli /iscsi create "$IQN"
    # Run it in demo mode, i.e. no authentication
    targetcli /iscsi/"$IQN"/tpg1 set attribute authentication=0 demo_mode_write_protect=0 generate_node_acls=1 cache_dynamic_acls=1

    # Create unique "block volume" (i.e. flat file) on the *host*.
    # Having it in the container confuses kernel from some reason
    # and it's not able to server multiple LUNs from different
    # containers.
    # /srv/iscsi must be bind-mount from the host.
    cp /block /srv/iscsi/"$IQN"

    # Make the block volume available through our IQN as LUN 0
    targetcli /backstores/fileio create block-"$IQN" /srv/iscsi/"$IQN"
    targetcli /iscsi/"$IQN"/tpg1/luns create /backstores/fileio/block-"$IQN"

    echo "iscsi target started"
}

function stop()
{
    echo "stopping iscsi target"
    # Remove IQN
    targetcli /iscsi/"$IQN"/tpg1/luns/ delete 0
    targetcli /iscsi delete "$IQN"
    # Remove block device mapping
    targetcli /backstores/fileio delete block-"$IQN"
    /bin/rm -f /srv/iscsi/"$IQN"
    echo "iscsi target stopped"
    exit 0
}


trap stop TERM
start

while true; do
    sleep 1
done
