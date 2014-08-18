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

# This file is meant to run on the master.  It takes the release in the current
# directory and installs everything that needs to be installed.  It will then
# also kick off a saltstack config pass

RELEASE_BASE=$(dirname $0)/../..

echo "Installing release files"

# Put all of the salt stuff under /srv
mkdir -p /srv
cp -R --preserve=mode $RELEASE_BASE/src/saltbase/* /srv

# Copy various go source code into the right places in the salt directory
# hieararchy so it can be downloaded/built on all the nodes.
mkdir -p /srv/salt/apiserver/go
cp -R --preserve=mode $RELEASE_BASE/src/go/* /srv/salt/apiserver/go

mkdir -p /srv/salt/kube-proxy/go
cp -R --preserve=mode $RELEASE_BASE/src/go/* /srv/salt/kube-proxy/go

mkdir -p /srv/salt/controller-manager/go
cp -R --preserve=mode $RELEASE_BASE/src/go/* /srv/salt/controller-manager/go

mkdir -p /srv/salt/scheduler/go
cp -R --preserve=mode $RELEASE_BASE/src/go/* /srv/salt/scheduler/go

mkdir -p /srv/salt/kubelet/go
cp -R --preserve=mode $RELEASE_BASE/src/go/* /srv/salt/kubelet/go

