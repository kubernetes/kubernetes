#!/bin/bash

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

set -e

# gencerts.sh generates the certificates for the CRL tests.
# 
# It is not expected to be run often (there is no go generate rule), and mainly
# exists for documentation purposes.

# clean up any previous runs
rm -f *.crt
rm -f *.key
rm -f *.crl

cat > client.conf << EOF
[req]
req_extensions = v3_req
distinguished_name = req_distinguished_name
[req_distinguished_name]
[ v3_req ]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth
EOF

# Create a certificate authority
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -days 100000 -out ca.crt -subj "/CN=crl_ca"

# Create 3 different client certiticates
for i in `seq 0 1 2`; do
    openssl genrsa -out client_${i}.key 2048
    openssl req -new -key client_${i}.key -out client_${i}.csr \
		-subj "/CN=crl_client" -config client.conf
    openssl x509 -req -in client_${i}.csr -CA ca.crt \
		-CAkey ca.key -CAcreateserial -out client_${i}.crt \
		-days 100000 -extensions v3_req -extfile client.conf
done

# Each CRL will revoke the certificates with the same ID.
for i in `seq 0 1 2`; do
  echo "0$i" > crl_number.conf
  rm -f crl.db
  touch crl.db
  
  cat > crl.conf << EOF
[ ca ]
default_ca	= CA_default		# The default ca section

[ CA_default ]
database = crl.db
crlnumber = crl_number.conf


default_days	= 10000			# how long to certify for
default_crl_days= 10000			# how long before next CRL
default_md	= default		# use public key default MD


[ crl_ext ]
# CRL extensions.
# Only issuerAltName and authorityKeyIdentifier make any sense in a CRL.
# issuerAltName=issuer:copy
authorityKeyIdentifier=keyid:always,issuer:always
EOF

  openssl ca -gencrl -keyfile ca.key -cert ca.crt -out crl_${i}.crl -config crl.conf
  openssl ca -revoke client_${i}.crt -keyfile ca.key -cert ca.crt -config crl.conf
  openssl ca -gencrl -keyfile ca.key -cert ca.crt -out crl_${i}.crl -config crl.conf
done

# Clean up after we're done.
rm *.csr
rm ca.*
rm *.conf
rm *.old
rm *.db
rm *.db.attr
