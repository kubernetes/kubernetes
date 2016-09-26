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

for i in `seq 0 1 1`; do
  # Create a certificate authority
  openssl genrsa -out ca_ca${i}.key 2048
  openssl req -x509 -new -nodes -key ca_ca${i}.key -days 100000 -out ca_ca${i}.crt \
	  -subj "/CN=crl_ca_${i}/SN=8/O=example/OU=foobar/L=London/OU=group ${i}"
done

# Create 2 different client certiticates for each cert
for i in `seq 0 1 1`; do
  for j in `seq 0 1 1`; do
    openssl genrsa -out client_ca${i}_id${j}.key 2048
    openssl req -new -key client_ca${i}_id${j}.key -out client_ca${i}_id${j}.csr \
      -subj "/CN=crl_client" -config client.conf
    openssl x509 -req -in client_ca${i}_id${j}.csr -CA ca_ca${i}.crt \
      -CAkey ca_ca${i}.key -CAcreateserial -out client_ca${i}_id${j}.crt \
      -days 100000 -extensions v3_req -extfile client.conf \
      -set_serial ${j} # set the serial number for the certificate
  done
done

# Each CRL will revoke the certificates with the same ID.
for i in `seq 0 1 1`; do
  for j in `seq 0 1 1`; do
    echo "$i$j" > crl_number.conf
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
  
    openssl ca -gencrl -keyfile ca_ca${i}.key -cert ca_ca${i}.crt -out crl_ca${i}_id${j}.crl -config crl.conf
	openssl ca -revoke client_ca${i}_id${j}.crt -keyfile ca_ca${i}.key -cert ca_ca${i}.crt -config crl.conf
	openssl ca -gencrl -keyfile ca_ca${i}.key -cert ca_ca${i}.crt -out crl_ca${i}_id${j}.crl -config crl.conf
  done
done

# Clean up after we're done.
rm *.csr
rm *.key
rm ca_*
rm *.conf
rm *.old
rm *.db
rm *.db.attr
