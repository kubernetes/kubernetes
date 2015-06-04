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

script_dir=$(cd "$(dirname ${BASH_SOURCE})" && pwd -P)
source "${script_dir}/util-temp-dir.sh"

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
  local certdir="$1"

  openssl req -nodes -newkey rsa:2048 \
    -keyout "${certdir}/root-ca.key" \
    -out "${certdir}/root-ca.csr" \
    -subj "/C=GB/ST=London/L=London/O=example/OU=IT/CN=example.com"

  openssl x509 -req -days 3650 \
    -in "${certdir}/root-ca.csr" \
    -out "${certdir}/root-ca.crt" \
    -signkey "${certdir}/root-ca.key"
}

# Creates an apiserver key and certificate with the given IPs & kubernetes.* domain names.
# Uses the current dir for scratch work.
function cluster::mesos::docker::create_apiserver_cert_inner {
  local in_dir="$1" # must contain root-ca.crt & root-ca.key
  local out_dir="$2"
  local apiserver_ip="$3"
  local service_ip="$4"
  local workspace="$(pwd)"

  mkdir -p "${out_dir}"

  local OPENSSL_CNF=$(cluster::mesos::docker::find_openssl_config)

  # create apiserver key and certificate sign request
  local SANS="IP:${apiserver_ip},IP:${service_ip},DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.cluster.local"
  openssl req -nodes -newkey rsa:2048 \
    -keyout "${workspace}/apiserver.key" -out "${workspace}/apiserver.csr" \
    -reqexts SAN -config <(cat "${OPENSSL_CNF}"; echo -e "[SAN]\nsubjectAltName=$SANS") \
    -subj "/C=GB/ST=London/L=London/O=example/OU=IT/CN=example.com"

  # sign with root-ca
  mkdir -p ${workspace}/demoCA/newcerts
  touch ${workspace}/demoCA/index.txt
  echo 1000 > ${workspace}/demoCA/serial
  openssl ca -cert "${in_dir}/root-ca.crt" -keyfile "${in_dir}/root-ca.key" \
    -batch -days 3650 -in "${workspace}/apiserver.csr" \
    -config <(sed 's/.*\(copy_extensions = copy\)/\1/' ${OPENSSL_CNF}) >/dev/null

  # check certificate for subjectAltName extension
  if ! openssl x509 -in "${workspace}/demoCA/newcerts/1000.pem" -text -noout | grep -q kubernetes.default.svc.cluster.local; then
    echo "ERROR: openssl failed to add subjectAltName extension" 1>&2
    return 1
  fi

  # write to out_dir
  cp "${workspace}/demoCA/newcerts/1000.pem" "${out_dir}/apiserver.crt"
  cp "${workspace}/apiserver.key" "${out_dir}/"
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
  openssl genrsa -out "${key_file_path}" 2048
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