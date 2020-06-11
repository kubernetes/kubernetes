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
    cat <<EOF > /etc/tgt/conf.d/kubernetes.conf
default-driver iscsi

<target $IQN>
        backing-store /block
</target>
EOF
    # Using -f (foreground) to print logs to stdout/stderr
    tgtd -d1 -f &
    echo "iscsi target started"
}

function stop()
{
    echo "stopping iscsi target"
    tgtadm --op update --mode sys --name State -v offline
    tgt-admin --update ALL -c /dev/null
    tgtadm --op delete --mode system
    sleep 1
    killall tgtd
    exit 0
}


trap stop TERM
start

while true; do
    sleep 1
done
