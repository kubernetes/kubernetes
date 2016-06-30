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

##########################################################
#
# These parameters describe objects we are using from
# Photon Controller. They are all assumed to be pre-existing.
#
# Note: if you want help in creating them, you can use 
# the setup-prereq.sh script, which will create any of these
# that do not already exist.
#
##########################################################

# Pre-created tenant for Kubernetes to use
PHOTON_TENANT=kube-tenant

# Pre-created project in PHOTON_TENANT for Kubernetes to use
PHOTON_PROJECT=kube-project

# Pre-created VM flavor for Kubernetes master to use
# Can be same as master
# We recommend at least 1GB of memory
PHOTON_MASTER_FLAVOR=kube-vm

# Pre-created VM flavor for Kubernetes node to use
# Can be same as master
# We recommend at least 2GB of memory
PHOTON_NODE_FLAVOR=kube-vm

# Pre-created disk flavor for Kubernetes to use
PHOTON_DISK_FLAVOR=kube-disk

# Pre-created Debian 8 image with kube user uploaded to Photon Controller
# Note: While Photon Controller allows multiple images to have the same
# name, we assume that there is exactly one image with this name.
PHOTON_IMAGE=kube

##########################################################
# 
# Parameters just for the setup-prereq.sh script: not used
# elsewhere. If you create the above objects by hand, you 
# do not need to edit these. 
# 
# Note that setup-prereq.sh also creates the objects 
# above.
# 
##########################################################

# The specifications for the master and node flavors
SETUP_MASTER_FLAVOR_SPEC="vm 1 COUNT, vm.cpu 1 COUNT, vm.memory 2 GB"
SETUP_NODE_FLAVOR_SPEC=${SETUP_MASTER_FLAVOR_SPEC}

# The specification for the ephemeral disk flavor. 
SETUP_DISK_FLAVOR_SPEC="ephemeral-disk 1 COUNT"

# The specification for the tenant resource ticket and the project resources
SETUP_TICKET_SPEC="vm.memory 1000 GB, vm 1000 COUNT"
SETUP_PROJECT_SPEC="${SETUP_TICKET_SPEC}"
