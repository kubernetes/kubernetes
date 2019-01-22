#!/usr/bin/env bash

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

#
# Configures and launches a new MON.
#

# monitor setup
monmaptool --create --clobber --fsid `uuidgen` --add a $1:6789 /etc/ceph/monmap
mkdir /var/lib/ceph/mon/ceph-a
ceph-mon -i a --mkfs --monmap /etc/ceph/monmap -k /var/lib/ceph/mon/keyring
cp /var/lib/ceph/mon/keyring /var/lib/ceph/mon/ceph-a
ceph-mon -i a --monmap /etc/ceph/monmap -k /var/lib/ceph/mon/ceph-a/keyring

# client setup (handy)
cp /var/lib/ceph/mon/keyring /etc/ceph

# for this test we want to
ceph osd getcrushmap -o /tmp/crushc
crushtool -d /tmp/crushc -o /tmp/crushd
sed -i 's/step chooseleaf firstn 0 type host/step chooseleaf firstn 0 type osd/' /tmp/crushd
crushtool -c /tmp/crushd -o /tmp/crushc
ceph osd setcrushmap -i /tmp/crushc
