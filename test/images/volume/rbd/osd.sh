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
# Configures and launches a new OSD.
#

ceph osd create
ceph-osd -i $1 --mkfs --mkkey
ceph auth add osd.$1 osd 'allow *' mon 'allow rwx' -i /var/lib/ceph/osd/ceph-$1/keyring
ceph osd crush add $1 1 root=default host=cephbox
ceph-osd -i $1 -k /var/lib/ceph/osd/ceph-$1/keyring
