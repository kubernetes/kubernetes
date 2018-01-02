#!/bin/bash

if ! [[ "$0" =~ "./gencerts.sh" ]]; then
	echo "must be run from 'fixtures'"
	exit 255
fi

if ! which cfssl; then
	echo "cfssl is not installed"
	exit 255
fi

cfssl gencert --initca=true ./ca-csr.json | cfssljson --bare ./ca
mv ca.pem ca.crt

cfssl gencert \
    --ca ./ca.crt \
    --ca-key ./ca-key.pem \
    --config ./gencert.json \
    ./server-ca-csr.json | cfssljson --bare ./server
mv server.pem server.crt
mv server-key.pem server.key.insecure

cfssl gencert --ca ./ca.crt \
    --ca-key ./ca-key.pem \
    --config ./gencert.json \
    ./server-ca-csr.json 2>revoked.stderr | cfssljson --bare ./server-revoked
mv server-revoked.pem server-revoked.crt
mv server-revoked-key.pem server-revoked.key.insecure

grep serial revoked.stderr | awk ' { print $9 } ' >revoke.txt
cfssl gencrl revoke.txt ca.crt ca-key.pem | base64 -d >revoke.crl

rm -f *.csr *.pem *.stderr *.txt
