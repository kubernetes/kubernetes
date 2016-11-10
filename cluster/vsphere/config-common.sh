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
# export GOVC_URL='hostname' # hostname of the vc
# export GOVC_USERNAME='username' # username for logging into the vsphere.
# export GOVC_PASSWORD='password' # password for the above username
# export GOVC_NETWORK='Network Name' # Name of the network the vms should join. Many times it could be "VM Network"
# export GOVC_DATASTORE='target datastore'
# To get resource pool via govc: govc ls -l 'host/*' | grep ResourcePool | awk '{print $1}' | xargs -n1 -t govc pool.info
# export GOVC_RESOURCE_POOL='resource pool or cluster with access to datastore'
# export GOVC_GUEST_LOGIN='kube:kube' # Used for logging into kube.vmdk during deployment.
# export GOVC_PORT=443 # The port to be used by vSphere cloud provider plugin
# To get datacente via govc: govc datacenter.info
# export GOVC_DATACENTER='ha-datacenter' # The datacenter to be used by vSphere cloud provider plugin
# export GOVC_GUEST_LOGIN='kube:kube' # Used for logging into kube.vmdk during deployment.

# Set GOVC_INSECURE if the host in GOVC_URL is using a certificate that cannot
# be verified (i.e. a self-signed certificate), but IS trusted.
# export GOVC_INSECURE=1
