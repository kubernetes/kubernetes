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

# This script is intended to set up the files necessary to run a master.
# It currently creates:
#  * The basic auth file for access to the kubernetes api server
#  * Service tokens for accessing the kubernetes api server
#  * The CA cert and keys for HTTPS access to the kubernetes api server
set -o errexit
set -o nounset
set -o pipefail

create_token() {
  echo $(cat /dev/urandom | base64 | tr -d "=+/" | dd bs=32 count=1 2> /dev/null)
}

# Additional address of the API server to be added to the
# list of Subject Alternative Names of the server TLS certificate
# Should contain internal IP, i.e. IP:10.0.0.1 for 10.0.0.0/24 cluster IP range
EXTRA_SANS=$1
DATA_DIR=/srv/kubernetes

# Files in /data are persistent across reboots, so we don't want to re-create the files if they already
# exist, because the state is persistent in etcd too, and we don't want a conflict between "old" data in
# etcd and "new" data that this script would create for apiserver. Therefore, if the file exist, skip it.
if [[ ! -f ${DATA_DIR}/ca.crt ]]; then

	# Create HTTPS certificates
	groupadd -f -r kube-cert

	# hostname -I gets the ip of the node
	/make-ca-cert.sh $(hostname -I | awk '{print $1}') ${EXTRA_SANS}

	echo "Certificates created $(date)"
else
	echo "Certificates already found, not recreating."
fi

if [[ ! -f ${DATA_DIR}/basic_auth.csv ]]; then

	# Create basic token authorization
	echo "admin,admin,admin" > ${DATA_DIR}/basic_auth.csv

	echo "basic_auth.csv created $(date)"
else
	echo "basic_auth.csv already found, not recreating."
fi

if [[ ! -f ${DATA_DIR}/known_tokens.csv ]]; then

	# Create known tokens for service accounts
	echo "$(create_token),admin,admin" >> ${DATA_DIR}/known_tokens.csv
	echo "$(create_token),kubelet,kubelet" >> ${DATA_DIR}/known_tokens.csv
	echo "$(create_token),kube_proxy,kube_proxy" >> ${DATA_DIR}/known_tokens.csv

	echo "known_tokens.csv created $(date)"
else
	echo "known_tokens.csv already found, not recreating."
fi

while true; do
  sleep 3600
done
