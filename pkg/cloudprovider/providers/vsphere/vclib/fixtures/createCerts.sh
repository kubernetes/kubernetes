#!/usr/bin/env bash

set -eu

# 1. create CA key
# 2. create CA cert
# 3. create CSR with IP SAN
# 4. create server cert with IP SAN

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
