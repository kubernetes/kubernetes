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

# TODO: this isn't quite piped into all the right places...
ZONE=us-west-2
MASTER_SIZE=t2.micro
MINION_SIZE=t2.micro
NUM_MINIONS=4

# This is the ubuntu 14.04 image for us-west-2 + ebs
# See here: http://cloud-images.ubuntu.com/locator/ec2/ for other images
# This will need to be updated from time to time as amis are deprecated
IMAGE=ami-39501209
INSTANCE_PREFIX=kubernetes
AWS_SSH_KEY=$HOME/.ssh/kube_aws_rsa

MASTER_NAME="ip-172-20-0-9.$ZONE.compute.internal"
MASTER_TAG="${INSTANCE_PREFIX}-master"
MINION_TAG="${INSTANCE_PREFIX}-minion"
MINION_NAMES=($(eval echo ip-172-20-0-1{0..$(($NUM_MINIONS-1))}.$ZONE.compute.internal))
MINION_IP_RANGES=($(eval echo "10.244.{1..${NUM_MINIONS}}.0/24"))
MINION_SCOPES=""
POLL_SLEEP_INTERVAL=3
PORTAL_NET="10.0.0.0/16"

# Optional: Install node logging
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=elasticsearch # options: elasticsearch, gcp

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING=false
ELASTICSEARCH_LOGGING_REPLICAS=1

IAM_PROFILE="kubernetes"
LOG="/dev/null"

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="kubernetes.local"
DNS_REPLICAS=1
