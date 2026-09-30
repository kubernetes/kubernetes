#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Requires smallstep step CLI 0.30.6. These commands sign locally without a step-ca server.
# Unencrypted keys and long lifetimes are only suitable for test fixtures.
flags=(--kty EC --curve P-256 --no-password --insecure --force --not-after 876000h)

step certificate create Client-CA client-ca.pem client-ca-key.pem \
  --template generate.client-ca.json "${flags[@]}"
step certificate create Server-CA server-ca.pem server-ca-key.pem \
  --template generate.server-ca.json "${flags[@]}"
step certificate create "My Client" client.pem client-key.pem \
  --ca client-ca.pem --ca-key client-ca-key.pem --template generate.client.json "${flags[@]}"
step certificate create test-service2.test-ns.svc server.pem server-key.pem \
  --ca server-ca.pem --ca-key server-ca-key.pem --template generate.server.json "${flags[@]}"
