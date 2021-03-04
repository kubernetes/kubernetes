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
# Configures /etc/ceph.conf from a template.
#

echo "
[global]
auth cluster required = none
auth service required = none
auth client required = none

[mon.a]
host = cephbox
mon addr = $1

[osd]
osd journal size = 128
journal dio = false

# allow running on ext4
osd max object name len = 256
osd max object namespace len = 64

[osd.0]
osd host = cephbox
" > /etc/ceph/ceph.conf

