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

cfssl gencert -initca generate.client-ca.json | cfssljson -bare client-ca
cfssl gencert -initca generate.server-ca.json | cfssljson -bare server-ca

cfssl gencert -ca client-ca.pem -ca-key client-ca-key.pem -config generate.profiles.json --profile=client generate.client.json | cfssljson -bare client
cfssl gencert -ca server-ca.pem -ca-key server-ca-key.pem -config generate.profiles.json --profile=server generate.server.json | cfssljson -bare server

rm ./*.csr
