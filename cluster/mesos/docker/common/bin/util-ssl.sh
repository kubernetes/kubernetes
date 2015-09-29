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

# Sourcable SSL functions

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

bin="$(cd "$(dirname "${BASH_SOURCE}")" && pwd -P)"
source "${bin}/util-temp-dir.sh"

function cluster::mesos::docker::find_openssl_config {
  for candidate in "/etc/ssl/openssl.cnf" "/System/Library/OpenSSL/openssl.cnf"; do
    if [ -f "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "ERROR: cannot find openssl.cnf" 1>&2
  return 1
}

function cluster::mesos::docker::create_root_certificate_authority {
  local out_dir="$1"

  # TODO(karlkfi): extract config
  local subject="/C=GB/ST=London/L=London/O=example/OU=IT/CN=example.com"

  echo "Creating private key" 1>&2
  openssl genrsa -out "${out_dir}/root-ca.key" 2048

  echo "Creating certificate sign request" 1>&2
  openssl req -nodes -new -utf8 \
    -key "${out_dir}/root-ca.key" \
    -out "${out_dir}/root-ca.csr" \
    -subj "${subject}"

  echo "Signing new certificate with private key" 1>&2
  openssl x509 -req -days 3650 \
    -in "${out_dir}/root-ca.csr" \
    -out "${out_dir}/root-ca.crt" \
    -signkey "${out_dir}/root-ca.key"

  echo "Key: ${out_dir}/root-ca.key" 1>&2
  echo "Cert: ${out_dir}/root-ca.crt" 1>&2
}

# Creates an apiserver key and certificate with the given IPs & kubernetes.* domain names.
# Uses the current dir for scratch work.
function cluster::mesos::docker::create_apiserver_cert_inner {
  local in_dir="$1"
  local out_dir="$2"
  local apiserver_ip="$3"
  local service_ip="$4"
  local workspace="$(pwd)"

  if [ ! -f "${in_dir}/root-ca.key" ]; then
    echo "Signing key not found: ${in_dir}/root-ca.key"
    return 1
  fi
  if [ ! -f "${in_dir}/root-ca.crt" ]; then
    echo "Root certificate not found: ${in_dir}/root-ca.key"
    return 1
  fi

  mkdir -p "${out_dir}"

  local openssl_cnf=$(cluster::mesos::docker::find_openssl_config)

  # TODO(karlkfi): extract config
  local subject="/C=GB/ST=London/L=London/O=example/OU=IT/CN=example.com"
  local cluster_domain="cluster.local"
  local service_name="kubernetes"
  local service_namespace="default"
  local subject_alt_name="IP:${apiserver_ip},IP:${service_ip},DNS:${service_name},DNS:${service_name}.${service_namespace},DNS:${service_name}.${service_namespace}.svc,DNS:${service_name}.${service_namespace}.svc.${cluster_domain}"

  echo "Creating private key" 1>&2
  openssl genrsa -out "${workspace}/apiserver.key" 2048

  echo "Creating certificate sign request" 1>&2
  openssl req -nodes -new -utf8 \
    -key "${workspace}/apiserver.key" \
    -out "${workspace}/apiserver.csr" \
    -reqexts SAN \
    -config <(cat "${openssl_cnf}"; echo -e "[SAN]\nsubjectAltName=${subject_alt_name}") \
    -subj "${subject}"

  echo "Validating certificate sign request" 1>&2
  openssl req -text -noout -in "${workspace}/apiserver.csr" | grep -q "${service_name}.${service_namespace}.svc.${cluster_domain}"

  echo "Signing new certificate with root certificate authority key" 1>&2
  mkdir -p "${workspace}/demoCA/newcerts"
  touch "${workspace}/demoCA/index.txt"
  echo 1000 > "${workspace}/demoCA/serial"
  openssl ca -batch \
    -days 3650 \
    -in "${workspace}/apiserver.csr" \
    -cert "${in_dir}/root-ca.crt" \
    -keyfile "${in_dir}/root-ca.key" \
    -config <(sed 's/.*\(copy_extensions = copy\)/\1/' ${openssl_cnf})

  echo "Validating signed certificate" 1>&2
  openssl x509 -in "${workspace}/demoCA/newcerts/1000.pem" -text -noout | grep -q "${service_name}.${service_namespace}.svc.${cluster_domain}"

  echo "Key: ${out_dir}/apiserver.key" 1>&2
  cp "${workspace}/apiserver.key" "${out_dir}/apiserver.key"

  echo "Cert: ${out_dir}/apiserver.crt" 1>&2
  cp "${workspace}/demoCA/newcerts/1000.pem" "${out_dir}/apiserver.crt"
}

# Creates an apiserver key and certificate with the given IPs & kubernetes.* domain names.
function cluster::mesos::docker::create_apiserver_cert {
  local in_dir="$1" # must contain root-ca.crt & root-ca.key
  local out_dir="$2"
  local apiserver_ip="$3"
  local service_ip="$4"

  cluster::mesos::docker::run_in_temp_dir "k8sm-certs" \
    "cluster::mesos::docker::create_apiserver_cert_inner" \
      "${in_dir}" "${out_dir}" "${apiserver_ip}" "${service_ip}"
}

# Creates an rsa key (for signing service accounts).
function cluster::mesos::docker::create_rsa_key {
  local key_file_path="$1"

  # buffer output until failure
  local output=$((
    openssl genrsa -out "${key_file_path}" 2048 || exit $?
  ) 2>&1)
  local exit_status="$?"
  if [ "${exit_status}" != 0 ]; then
    echo "${output}" 1>&2
    return "${exit_status}"
  fi

  echo "Key: ${key_file_path}" 1>&2
}

# Creates a k8s token auth user file.
# See /docs/admin/authentication.md
function cluster::mesos::docker::create_token_user {
  local user_name="$1"
  echo "$(openssl rand -hex 32),${user_name},${user_name}"
}

# Creates a k8s basic auth user file.
# See /docs/admin/authentication.md
function cluster::mesos::docker::create_basic_user {
  local user_name="$1"
  local password="$2"
  echo "${password},${user_name},${user_name}"
}
