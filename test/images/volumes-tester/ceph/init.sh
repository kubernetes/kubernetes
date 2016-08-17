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

#set -e
set -x

# clean up
rm -f /etc/ceph/*

pkill -9 ceph-mon
pkill -9 ceph-osd
pkill -9 ceph-mds

mkdir -p /var/lib/ceph
mkdir -p /var/lib/ceph/osd
mkdir -p /var/lib/ceph/osd/ceph-0

# create hostname for ceph monitor
MASTER=`hostname -s`

ip=$(ip -4 -o a | grep eth0 | awk '{print $4}' | cut -d'/' -f1)
echo "$ip $MASTER" >> /etc/hosts

#create ceph cluster
ceph-deploy --overwrite-conf new ${MASTER}  
ceph-deploy --overwrite-conf mon create-initial ${MASTER}
ceph-deploy --overwrite-conf mon create ${MASTER}
ceph-deploy  gatherkeys ${MASTER}  

# set osd  params for minimal configuration
echo "osd crush chooseleaf type = 0" >> /etc/ceph/ceph.conf
echo "osd journal size = 100" >> /etc/ceph/ceph.conf
echo "osd pool default size = 1" >> /etc/ceph/ceph.conf
echo "osd pool default pgp num = 8" >> /etc/ceph/ceph.conf
echo "osd pool default pg num = 8" >> /etc/ceph/ceph.conf

/sbin/service ceph -c /etc/ceph/ceph.conf stop mon.${MASTER}
/sbin/service ceph -c /etc/ceph/ceph.conf start mon.${MASTER}

# create ceph osd
ceph osd create
ceph-osd -i 0 --mkfs --mkkey
ceph auth add osd.0 osd 'allow *' mon 'allow rwx' -i /var/lib/ceph/osd/ceph-0/keyring
ceph osd crush add 0 1 root=default host=${MASTER}
ceph-osd -i 0 -k /var/lib/ceph/osd/ceph-0/keyring

#see if we are ready to go  
ceph osd tree  

# create ceph fs
ceph osd pool create cephfs_data 4
ceph osd pool create cephfs_metadata 4
ceph fs new cephfs cephfs_metadata cephfs_data
ceph-deploy --overwrite-conf mds create ${MASTER}

# uncomment the following for rbd test
# ceph osd pool create kube 4
# rbd create foo --size 10 --pool kube

ps -ef |grep ceph
ceph osd dump

# add new client with a pre defined keyring
# this keyring must match the ceph secret in e2e test
cat > /etc/ceph/ceph.client.kube.keyring <<EOF
[client.kube]
        key = AQAMgXhVwBCeDhAA9nlPaFyfUSatGD4drFWDvQ==
        caps mds = "allow rwx"
        caps mon = "allow rwx"
        caps osd = "allow rwx"
EOF
ceph auth import -i /etc/ceph/ceph.client.kube.keyring

# mount it through ceph-fuse and copy file to ceph fs
ceph-fuse -m ${MASTER}:6789 /mnt
cp /tmp/index.html /mnt
chmod 644 /mnt/index.html

# watch
ceph -w
