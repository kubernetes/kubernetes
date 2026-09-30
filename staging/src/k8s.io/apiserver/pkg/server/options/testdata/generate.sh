#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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
flags=(--kty EC --curve P-256 --no-password --insecure --force)

step certificate create Root-CA root.pem root-key.pem \
  --template root.template.json --not-after 876000h "${flags[@]}"
step certificate create Intermediate-CA intermediate.pem intermediate-key.pem \
  --ca root.pem --ca-key root-key.pem --template intermediate.template.json \
  --not-after 876000h "${flags[@]}"
step certificate create "My Client" client-valid.pem client-valid-key.pem \
  --ca intermediate.pem --ca-key intermediate-key.pem --template client.template.json \
  --not-after 876000h "${flags[@]}"
# Keep the zero-length validity interval used by the expired-certificate tests.
step certificate create "My Client" client-expired.pem client-expired-key.pem \
  --ca intermediate.pem --ca-key intermediate-key.pem --template client.template.json \
  --not-before 1990-12-31T23:59:00Z --not-after 1990-12-31T23:59:00Z "${flags[@]}"
