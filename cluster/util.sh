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

# A library of helper functions and constant for the local config.

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
source $(dirname ${BASH_SOURCE})/${KUBE_CONFIG_FILE-"config-default.sh"}

# Find the release to use.  If passed in, go with that and validate.  If not use
# the release/config.sh version assuming a dev workflow.
function find-release() {
  if [ -n "$1" ]; then
    RELEASE_NORMALIZED=$1
  else
    local RELEASE_CONFIG_SCRIPT=$(dirname $0)/../release/config.sh
    if [ -f $(dirname $0)/../release/config.sh ]; then
      . $RELEASE_CONFIG_SCRIPT
      normalize_release
    fi
  fi

  # Do one final check that we have a good release
  if ! gsutil -q stat $RELEASE_NORMALIZED/master-release.tgz; then
    echo "Could not find release tar.  If developing, make sure you have run src/release/release.sh to create a release."
    exit 1
  fi
  echo "Release: ${RELEASE_NORMALIZED}"
}

# Use the gcloud defaults to find the project.  If it is already set in the
# environment then go with that.
function detect-project () {
  if [ -z "$PROJECT" ]; then
    PROJECT=$(gcloud config list project | tail -n 1 | cut -f 3 -d ' ')
  fi

  if [ -z "$PROJECT" ]; then
    echo "Could not detect Google Cloud Platform project.  Set the default project using 'gcloud config set project <PROJECT>'"
    exit 1
  fi
  echo "Project: $PROJECT (autodetected from gcloud config)"
}

function detect-minions () {
  KUBE_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_ip=$(gcutil listinstances --format=csv --sort=external-ip \
      --columns=external-ip --filter="name eq ${MINION_NAMES[$i]}" \
      | tail -n 1)
    echo "Found ${MINION_NAMES[$i]} at ${minion_ip}"
    KUBE_MINION_IP_ADDRESSES+=("${minion_ip}")
  done
  if [ -z "$KUBE_MINION_IP_ADDRESSES" ]; then
    echo "Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi
}

function detect-master () {
  KUBE_MASTER=${MASTER_NAME}
  if [ -z "$KUBE_MASTER_IP" ]; then
    KUBE_MASTER_IP=$(gcutil listinstances --format=csv --sort=external-ip \
      --columns=external-ip --filter="name eq ${MASTER_NAME}" \
      | tail -n 1)
  fi
  if [ -z "$KUBE_MASTER_IP" ]; then
    echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)"
}

function get-password {
  file=${HOME}/.kubernetes_auth
  if [ -e ${file} ]; then
    user=$(cat $file | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
    passwd=$(cat $file | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
    return
  fi
  user=admin
  passwd=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')

  # Store password for reuse.
  cat << EOF > ~/.kubernetes_auth
{
  "User": "$user",
  "Password": "$passwd"
}
EOF
  chmod 0600 ~/.kubernetes_auth
}
