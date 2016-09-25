#!/usr/bin/env bash

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

# Azure location to deploy to. (Storage account, resource group, resources)
# Must be be specified in the compact form. ("westus" is ok, "West US" is not)
AZURE_LOCATION="${AZURE_LOCATION:-"westus"}"

# An identifier for the deployment. It can be left blank and an identifier
# will be generated from the date/time.
AZURE_DEPLOY_ID="${AZURE_DEPLOY_ID:-"kube-$(date +"%Y%m%d-%H%M%S")"}"

AZURE_MASTER_SIZE="${AZURE_MASTER_SIZE:-"Standard_A1"}"
AZURE_NODE_SIZE="${AZURE_NODE_SIZE:-"Standard_A1"}"

# Username of the admin account created on the VMs
AZURE_USERNAME="${AZURE_USERNAME:-"kube"}"

# Initial number of worker nodes to provision
NUM_NODES=${NUM_NODES:-3}

# The target Azure subscription ID
# This should be a GUID.
AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"

# The authentication mechanism to use. The default "device" is recommended as
# it requires the least ahead-of-time setup.
# This should be one of: { "device", "client_secret" }
AZURE_AUTH_METHOD="${AZURE_AUTH_METHOD:-"device"}"
