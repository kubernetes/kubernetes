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

set -eu

createKey() {
  openssl genrsa -out "$1" 4096
}

createCaCert() {
  openssl req -x509 \
    -subj "/C=US/ST=CA/O=Acme, Inc./CN=someCA" \
    -new -nodes -key "$2" -sha256 -days 10240 -out "$1"
}

createCSR() {
  openssl req -new -sha256 \
    -key "$2" \
    -subj "/C=US/ST=CA/O=Acme, Inc./CN=localhost" \
    -reqexts SAN \
    -config <(
      cat /etc/ssl/openssl.cnf ; \
      printf '\n[SAN]\n' ; \
      getSAN
      ) \
    -out "$1"
}

signCSR() {
  openssl x509 -req -in "$2" \
    -CA "$3" \
    -CAkey "$4" \
    -CAcreateserial \
    -days 3650 -sha256 \
    -extfile <( getSAN ) \
    -out "$1"
}

getSAN() {
  printf "subjectAltName=DNS:localhost,IP:127.0.0.1"
}

main() {
  local caCertPath="./ca.pem"
  local caKeyPath="./ca.key"
  local serverCsrPath="./server.csr"
  local serverCertPath="./server.pem"
  local serverKeyPath="./server.key"

  createKey "$caKeyPath"
  createCaCert "$caCertPath" "$caKeyPath"
  createKey "$serverKeyPath"
  createCSR "$serverCsrPath" "$serverKeyPath"
  signCSR "$serverCertPath" "$serverCsrPath" "$caCertPath" "$caKeyPath"
}

main "$@"
