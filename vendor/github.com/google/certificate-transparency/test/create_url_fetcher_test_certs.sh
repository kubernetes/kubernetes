#!/usr/bin/env bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
. ${DIR}/generate_certs.sh

set -e

TARGET="${DIR}/testdata/urlfetcher_test_certs"
mkdir -p ${TARGET}

# TODO(alcutter): fix the cert/ca scripts to work if called from a different
# directory
cd ${DIR}

echo "Generating CA certs in ${TARGET}"
mkdir -p ${TARGET}/ca-hashes
ca_setup ${TARGET} ca false
make_ca_certs ${TARGET} ${TARGET}/ca-hashes ca openssl
cp ${TARGET}/01.pem ${TARGET}/ca-cert.pem

sed -e '/0.organizationName=Certificate/ a\
commonName=localhost' ${DIR}/precert.conf > ${TARGET}/localhost.conf
sed -e '/0.organizationName=Certificate/ a\
commonName=not-localhost' ${DIR}/precert.conf > ${TARGET}/not-localhost.conf
sed -e '/0.organizationName=Certificate/ a\
commonName=binky.example.com' ${DIR}/precert.conf > ${TARGET}/binky_example_com.conf
sed -e '/0.organizationName=Certificate/ a\
commonName=*.example.com' ${DIR}/precert.conf > ${TARGET}/star_example_com.conf
sed -e '/0.organizationName=Certificate/ a\
commonName=example.com' ${DIR}/precert.conf > ${TARGET}/example_com.conf
sed -e '/0.organizationName=Certificate/ a\
commonName=127.0.0.1' ${DIR}/precert.conf > ${TARGET}/127_0_0_1.conf

request_cert ${TARGET} "not-localhost" ${TARGET}/not-localhost.conf true
issue_cert ${TARGET} ca "not-localhost" ${TARGET}/not-localhost.conf simple false "not-localhost"

request_cert ${TARGET} "localhost" ${TARGET}/localhost.conf true
issue_cert ${TARGET} ca "localhost" ${TARGET}/localhost.conf simple false "localhost"

request_cert ${TARGET} "binky_example_com" ${TARGET}/binky_example_com.conf true
issue_cert ${TARGET} ca "binky_example_com" ${TARGET}/binky_example_com.conf simple false "binky_example_com"

request_cert ${TARGET} "star_example_com" ${TARGET}/star_example_com.conf true
issue_cert ${TARGET} ca "star_example_com" ${TARGET}/star_example_com.conf simple false "star_example_com"

request_cert ${TARGET} "example_com" ${TARGET}/example_com.conf true
issue_cert ${TARGET} ca "example_com" ${TARGET}/example_com.conf simple false "example_com"

request_cert ${TARGET} "127_0_0_1" ${TARGET}/127_0_0_1.conf true
issue_cert ${TARGET} ca "127_0_0_1" ${TARGET}/127_0_0_1.conf simple false "127_0_0_1"

