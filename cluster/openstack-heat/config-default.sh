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

## Contains configuration values for the Openstack cluster

# Stack name
STACK_NAME=${STACK_NAME:-KubernetesStack}

# Keypair for kubernetes stack
KUBERNETES_KEYPAIR_NAME=${KUBERNETES_KEYPAIR_NAME:-kubernetes_keypair}

# Kubernetes release tar file
KUBERNETES_RELEASE_TAR=${KUBERNETES_RELEASE_TAR:-kubernetes-server-linux-amd64.tar.gz}

NUMBER_OF_MINIONS=${NUMBER_OF_MINIONS-3}

MAX_NUMBER_OF_MINIONS=${MAX_NUMBER_OF_MINIONS:-3}

MASTER_FLAVOR=${MASTER_FLAVOR:-m1.medium}

MINION_FLAVOR=${MINION_FLAVOR:-m1.medium}

EXTERNAL_NETWORK=${EXTERNAL_NETWORK:-public}

SWIFT_SERVER_URL=${SWIFT_SERVER_URL:-}

# Flag indicates if new image must be created. If 'false' then image with IMAGE_ID will be used.
# If 'true' then new image will be created from file config-image.sh
CREATE_IMAGE=${CREATE_IMAGE:-true} # use "true" for devstack

# Flag indicates if image should be downloaded
DOWNLOAD_IMAGE=${DOWNLOAD_IMAGE:-true}

# Image id which will be used for kubernetes stack
IMAGE_ID=${IMAGE_ID:-f0f394b1-5546-4b68-b2bc-8abe8a7e6b8b}

# DNS server address
DNS_SERVER=${DNS_SERVER:-8.8.8.8}

# Public RSA key path
CLIENT_PUBLIC_KEY_PATH=${CLIENT_PUBLIC_KEY_PATH:-~/.ssh/id_rsa.pub}

# Max time period for stack provisioning. Time in minutes.
STACK_CREATE_TIMEOUT=${STACK_CREATE_TIMEOUT:-60}

# Enable Proxy, if true kube-up will apply your current proxy settings(defined by *_PROXY environment variables) to the deployment.
ENABLE_PROXY=${ENABLE_PROXY:-false}

# Per-protocol proxy settings.
FTP_PROXY=${FTP_PROXY:-}
HTTP_PROXY=${HTTP_PROXY:-}
HTTPS_PROXY=${HTTPS_PROXY:-}
SOCKS_PROXY=${SOCKS_PROXY:-}

# IPs and Domains that bypass the proxy.
NO_PROXY=${NO_PROXY:-}
