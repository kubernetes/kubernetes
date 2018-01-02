#!/usr/bin/env bash

CERT_DIR=$1
CT_KEY=$2
SERVER=${3:-"127.0.0.1:8888"}

. generate_certs.sh

if [ ! -e $CERT_DIR/ca-database ]
then
  echo "Initialise CA"
  ca_setup $CERT_DIR ca false
fi

make_cert $CERT_DIR test ca $SERVER false $CT_KEY

../cpp/client/ct extension_data --sct_token=$CERT_DIR/test-cert.proof \
    --tls_extension_data_out=/tmp/xx.pem

$OPENSSLDIR/apps/openssl s_server -serverinfo /tmp/xx.pem -cert /tmp/ct-ca/test-cert.pem -key /tmp/ct-ca/test-key.pem
