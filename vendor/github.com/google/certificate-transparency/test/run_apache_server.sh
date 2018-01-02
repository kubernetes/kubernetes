#!/usr/bin/env bash

HTTPD_DIR=$1
OPENSSL_DIR=$2
SERVER=${2:-"127.0.0.1:8888"}

. generate_certs.sh

../cpp/client/ct extension_data --sct_token=testdata/test-cert.proof \
    --tls_extension_data_out=testdata/test-cert-proof-extension.pem

ln -sf $HTTPD_DIR/modules/ssl/.libs/mod_ssl.so .
ln -sf $HTTPD_DIR/modules/arch/unix/.libs/mod_unixd.so .

cd testdata

mkdir -p logs

LD_LIBRARY_PATH=$HTTPD_DIR/srclib/apr/.libs:$OPENSSL_DIR/lib $HTTPD_DIR/.libs/httpd -f `pwd`/../httpd-devel.conf -d `pwd`

