#!/bin/sh

# Upload whatever cert the named server presents to the pilot log

set -e

export PYTHONPATH=${PYTHONPATH}:../python
SERVER=$1
PUBKEY=$2 # PEM encoded file
#CT_SERVER='ct.googleapis.com/pilot'
CT_SERVER='localhost:8888'
TMP=`mktemp /tmp/cert.XXXXXX`

openssl s_client -connect $SERVER:443 -showcerts < /dev/null | tee $TMP

if ./ct --ct_server=$CT_SERVER --logtostderr --ct_server_submission=$TMP --ct_server_public_key=$PUBKEY upload
then
    echo Done
else
    echo Try fixing the chain
    TMP2=`mktemp /tmp/cert.XXXXXX`
    ./fix-chain.py $TMP | tee $TMP2
    ./ct --ct_server=$CT_SERVER --logtostderr --ct_server_submission=$TMP2 --ct_server_public_key=$PUBKEY upload
    rm $TMP2
fi

rm $TMP

