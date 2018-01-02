#!/usr/bin/env bash

# This script is mostly so we can test the test script for a standalone server.
# <storage file> is the sqlite3 database for the log.
# <certificate hash directory> contains the OpenSSL hashes for the
# accepted root certs.

set -e

if [ "$OPENSSLDIR" != "" ]; then
  MY_OPENSSL="$OPENSSLDIR/apps/openssl"
  export LD_LIBRARY_PATH=$OPENSSLDIR:$LD_LIBRARY_PATH
fi

if [ ! $MY_OPENSSL ]; then
# Try to use the system OpenSSL
  MY_OPENSSL=openssl
fi

if [ $# -lt 1 ]
then
  echo "$0 <storage file>"
  exit 1
fi

STORAGE=$1
KEY="testdata/ct-server-key.pem"
shift 1

# if [ ! -e $HASH_DIR ]
# then
#   echo "$HASH_DIR doesn't exist, creating"
#   mkdir $HASH_DIR
#   CERT="`pwd`/testdata/ca-cert.pem"
#   hash=`$MY_OPENSSL x509 -in $CERT -hash -noout`
#   ln -s $CERT $HASH_DIR/$hash.0
# fi

export TSAN_OPTIONS=${TSAN_OPTIONS:-log_path=tsan_log suppressions=../cpp/tsan_suppressions external_symbolizer_path=/usr/bin/llvm-symbolizer-3.4}

../cpp/server/xjson-server --port=8888 --key=$KEY \
  --logtostderr=true \
  --guard_window_seconds=5 \
  --tree_signing_frequency_seconds=10 --leveldb_db=$STORAGE \
  $*
