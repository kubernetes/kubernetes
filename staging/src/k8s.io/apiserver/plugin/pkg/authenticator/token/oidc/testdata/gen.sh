#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

set -e

rm ./*.pem

for N in $(seq 1 3); do
    ssh-keygen -t rsa -b 2048 -f rsa_"$N".pem -N ''
done

for N in $(seq 1 3); do
    ssh-keygen -t ecdsa -b 521 -f ecdsa_"$N".pem -N ''
done

rm ./*.pub
