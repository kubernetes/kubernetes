#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
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

function fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

# wait_for_nfs blocks until the NFS service is registered with rpcbind, which is
# the point at which the server actually answers mount requests. rpc.nfsd can
# leave the kernel server unusable without reporting an error, so registration is
# checked instead of trusting the exit status alone.
function wait_for_nfs()
{
    local i
    for i in $(seq 60); do
        if /usr/sbin/rpcinfo -p 127.0.0.1 2>/dev/null | grep -qw nfs; then
            return 0
        fi
        sleep 1
    done
    return 1
}

function start()
{

    unset gid
    # accept "-G gid" option
    while getopts "G:" opt; do
        case ${opt} in
            G) gid=${OPTARG};;
            *):;;
        esac
    done
    shift $((OPTIND - 1))

    # prepare /etc/exports
    for i in "$@"; do
        # fsid=0: needed for NFSv4
        echo "$i *(rw,fsid=0,insecure,no_root_squash)" >> /etc/exports
        if [ -v gid ] ; then
            chmod 070 "$i"
            chgrp "$gid" "$i"
        fi
        # move index.html to here
        /bin/cp /tmp/index.html "$i/"
        chmod 644 "$i/index.html"
        echo "Serving $i"
    done

    # start rpcbind if it is not started yet
    /usr/sbin/rpcinfo 127.0.0.1 > /dev/null; s=$?
    if [ $s -ne 0 ]; then
       echo "Starting rpcbind"
       /usr/sbin/rpcbind -w || fail "rpcbind failed to start"
    fi

    mount -t nfsd nfsd /proc/fs/nfsd || fail "failed to mount /proc/fs/nfsd"

    # -V 3: enable NFSv3
    /usr/sbin/rpc.mountd -V 3 || fail "rpc.mountd failed to start"

    /usr/sbin/exportfs -r || fail "failed to export the configured paths"
    # -G 10 to reduce grace time to 10 seconds (the lowest allowed)
    /usr/sbin/rpc.nfsd -G 10 -V 3 || fail "rpc.nfsd failed to start"
    /usr/sbin/rpc.statd --no-notify || fail "rpc.statd failed to start"

    # Only report readiness once the server really serves. Callers wait for the
    # message below before pointing workloads at this server, so printing it
    # while the server is down strands them on mounts that can never succeed.
    wait_for_nfs || fail "NFS service did not register with rpcbind"

    echo "NFS started"
}

function stop()
{
    echo "Stopping NFS"

    /usr/sbin/rpc.nfsd 0
    /usr/sbin/exportfs -au
    /usr/sbin/exportfs -f

    kill "$( pidof rpc.mountd )"
    umount /proc/fs/nfsd
    echo > /etc/exports
    exit 0
}

# rpc.statd has issues with very high ulimits
ulimit -n 65535

trap stop TERM

start "$@"

# Ugly hack to do nothing and wait for SIGTERM
while true; do
    sleep 5
done
