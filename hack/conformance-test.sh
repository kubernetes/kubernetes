#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# The conformance test is provided to let users run an e2e test
# against an already-setup cluster for which there is no automated
# setup, teardown, and other cluster/... scripts.
#
# The user must export these environment variables:
# KUBE_MASTER_IP to the ip address of the master.
# AUTH_CONFIG to the argument of the "--auth_config=" flag.
# If certs required, set CERT_DIR.
# 
# Example to test against a local vagrant cluster:
# declare -x AUTH_CONFIG="$HOME/.kubernetes_vagrant_auth"
# declare -x KUBE_MASTER_IP=10.245.1.2
# hack/conformance-test.sh
if [[ -z "KUBE_MASTER_IP" ]]; then
  echo "Must set KUBE_MASTER_IP before running conformance test."
  exit 1
fi
if [[ -z "AUTH_CONFIG" ]]; then
  echo "Must set AUTH_CONFIG before running conformance test."
  exit 1
fi
hack/ginkgo-e2e.sh
exit $?
