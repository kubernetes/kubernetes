#!/usr/bin/env bash

# Note: run make freebsd-links, make linux-links or make local-links
# before running this test

# TODO(ekasper): This test is getting too long. Split into smaller test cases.

source generate_certs.sh

PASSED=0
FAILED=0

if [ "$OPENSSLDIR" != "" ]; then
  MY_OPENSSL="$OPENSSLDIR/apps/openssl"
  export LD_LIBRARY_PATH=$OPENSSLDIR:$LD_LIBRARY_PATH
fi

if [ ! $MY_OPENSSL ]; then
# Try to use the system OpenSSL
  MY_OPENSSL=openssl
fi

echo Using `$MY_OPENSSL version`

test_connect() {
  cert_dir=$1
  hash_dir=$2
  log_server=$3
  ca=$4
  port=$5
  expect_fail=$6
  strict=$7
  audit=$8
 
  if [ "$audit" == "true" ]; then
    local client_flags="--ssl_client_ct_data_out=$cert_dir/$port.sct"
  fi

  # Continue tests on error
  set +e
  ../cpp/client/ct connect --ssl_server="127.0.0.1" --ssl_server_port=$port \
    --ct_server_public_key=$cert_dir/$log_server-key-public.pem \
    --ssl_client_trusted_cert_dir=$hash_dir --logtostderr=true \
    --ssl_client_require_sct=$strict \
    --ssl_client_expect_handshake_failure=$expect_fail $client_flags

  local retcode=$?
  set -e

  if [ $retcode -eq 0 ]; then
    echo "PASS"
    let PASSED=$PASSED+1
  else
    echo "FAIL"
    let FAILED=$FAILED+1
  fi
}

audit() {
  cert_dir=$1
  log_server=$2
  port=$3

  set +e
  ../cpp/client/ct audit --ct_server="127.0.0.1:8124" \
    --ct_server_public_key=$cert_dir/$log_server-key-public.pem \
    --ssl_client_ct_data_in=$cert_dir/$port.sct --logtostderr=true
  local retcode=$?
  set -e

  if [ $retcode -eq 0 ]; then
    echo "PASS"
    let PASSED=$PASSED+1
  else
    echo "FAIL"
    let FAILED=$FAILED+1
  fi
}

test_range() {
  ports=$1
  cert_dir=$2
  hash_dir=$3
  log_server=$4
  ca=$5
  conf=$6
  expect_fail=$7
  strict=$8
  audit=$9
  apache=${10}

  echo "Starting Apache"
  export APACHE_ENVVARS=./httpd.envvars # workaround Debian's apachectl
  $apache -d `pwd`/$cert_dir -f `pwd`/$conf -k start

  for port in $ports; do
    test_connect $cert_dir $hash_dir $log_server $ca $port $expect_fail \
      $strict $audit
  done

  echo "Stopping Apache"
  $apache -d `pwd`/$cert_dir -f `pwd`/$conf -k stop
  # Wait for Apache to die
  sleep 5

  if [ "$audit" == "true" ]; then
    echo "Starting audit"
    for port in $ports; do
      audit $cert_dir $log_server $port
    done
  fi
}

# Regression tests against known good/bad certificates
mkdir -p ca-hashes
hash=$($MY_OPENSSL x509 -in testdata/ca-cert.pem -hash -noout)
cp testdata/ca-cert.pem ca-hashes/$hash.0

echo "Testing known good/bad certificate configurations" 
mkdir -p testdata/logs

test_range "8125 8126 8127 8128 8129 8130" testdata ca-hashes ct-server ca \
  httpd-valid.conf false true false ./apachectl
test_range "8125 8126 8127 8128" testdata ca-hashes ct-server ca \
  httpd-invalid.conf false false false ./apachectl
test_range "8125 8126 8127 8128" testdata ca-hashes ct-server ca \
  httpd-invalid.conf true true false ./apachectl

rm -rf ca-hashes

# Generate new certs dynamically and repeat the test for valid certs
mkdir -p tmp
# A directory for trusted certs in OpenSSL "hash format"
mkdir -p tmp/ca-hashes

echo "Generating CA certificates in tmp and hashes in tmp/ca"
make_ca_certs `pwd`/tmp `pwd`/tmp/ca-hashes ca $MY_OPENSSL
ca_file="tmp/$ca-cert.pem"
echo "Generating log server keys in tmp"
make_log_server_keys `pwd`/tmp ct-server

# Start the log server and wait for it to come up
mkdir -p tmp/storage
mkdir -p tmp/storage/certs
mkdir -p tmp/storage/tree

test_ct_server() {
  flags=$@

  log_server_port=8124
  log_server_url=http://127.0.0.1:${log_server_port}

  # Set the tree signing frequency to 0 to ensure we sign as often as possible.
  echo "Starting CT server with trusted certs in $ca_file"
  ../cpp/server/ct-server --port=$log_server_port \
    --key="$cert_dir/$log_server-key.pem" \
    --trusted_cert_file="$ca_file" --logtostderr=true \
    --tree_signing_frequency_seconds=1 $flags &

  server_pid=$!
  sleep 2

  echo "Generating test certificates"
  make_cert `pwd`/tmp test ca $log_server_url false \
    `pwd`/tmp/ct-server-key-public.pem
  make_embedded_cert `pwd`/tmp test-embedded ca \
    $log_server_url false false `pwd`/tmp/ct-server-key-public.pem
  make_embedded_cert `pwd`/tmp test-embedded-with-preca \
    ca $log_server_url false true `pwd`/tmp/ct-server-key-public.pem
  # Generate a second set of certs that chain through an intermediate
  make_intermediate_ca_certs `pwd`/tmp intermediate ca

  make_cert `pwd`/tmp test-intermediate intermediate \
    $log_server_url true `pwd`/tmp/ct-server-key-public.pem
  make_embedded_cert `pwd`/tmp \
    test-embedded-with-intermediate intermediate $log_server_url \
    true false `pwd`/tmp/ct-server-key-public.pem
  make_embedded_cert `pwd`/tmp \
    test-embedded-with-intermediate-preca intermediate $log_server_url \
    true true `pwd`/tmp/ct-server-key-public.pem

  # Wait a bit to ensure the server signs the tree.
  sleep 5

  echo "Testing valid configurations with new certificates"
  mkdir -p tmp/logs
  test_range "8125 8126 8127 8128 8129 8130" tmp tmp/ca-hashes ct-server ca \
    httpd-valid.conf false true true ./apachectl

  # Stop the log server
  echo "Stopping CT server"
  kill -9 $server_pid  
  sleep 2
}

test_ct_server --sqlite_db=tmp/storage/ct

test_ct_server --cert_dir="tmp/storage/certs"  --tree_dir="tmp/storage/tree" \
    --cert_storage_depth=3 --tree_storage_depth=8

echo "Cleaning up"
rm -rf tmp
if [ $FAILED == 0 ]; then
  rm -rf testdata/logs
fi
echo "PASSED $PASSED tests"
echo "FAILED $FAILED tests"
