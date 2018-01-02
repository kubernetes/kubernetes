#!/usr/bin/env bash
# Starts a stand-alone mirror server.
# usage: run_mirror_server.sh <db_storage_location> [source_log_url] [source_log_public_key]

set -e
set -x

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
  echo "$0 <storage file> <target URL> <target public key PEM>"
  exit 1
fi

STORAGE=$1
TARGET=${2:-https://ct.googleapis.com/testtube}
KEY=${3:-../cloud/keys/testtube.pem}

export TSAN_OPTIONS=${TSAN_OPTIONS:-log_path=tsan_log suppressions=../cpp/tsan_suppressions external_symbolizer_path=/usr/bin/llvm-symbolizer-3.4}

../cpp/server/ct-mirror \
  --port=6962 \
  --target_public_key=$KEY \
  --target_log_uri=${TARGET} \
  --logtostderr=true \
  --leveldb_db=$STORAGE \
  --v 1 \
  --monitoring=prometheus \
  --etcd_servers=localhost:4001 \
  $*
