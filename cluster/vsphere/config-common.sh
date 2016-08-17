#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

SSH_OPTS="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR -C"

# These need to be set
#export GOVC_URL=
#export GOVC_DATACENTER=
#export GOVC_DATASTORE=
#export GOVC_NETWORK=
#export GOVC_GUEST_LOGIN=

# Set GOVC_INSECURE if the host in GOVC_URL is using a certificate that cannot
# be verified (i.e. a self-signed certificate), but IS trusted.
#export GOVC_INSECURE=1
