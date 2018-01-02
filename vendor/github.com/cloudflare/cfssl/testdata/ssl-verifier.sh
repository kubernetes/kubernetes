#!/bin/bash

KEY=$1
CRT=$2
IMM=$3

if [ "`cat $KEY | grep ENCRYPTED`" ]; then
    echo >&2 "Key is password-protected"
    exit 1
fi

KEYMOD=`openssl rsa -noout -modulus -in $KEY`
CRTMOD=`openssl x509 -noout -modulus -in $CRT`

if [ "$KEYMOD" != "$CRTMOD" ]; then
    echo >&2 "Key doesn't match the certificate"
    exit 1
fi

if [ -n "$IMM" ]; then
    cat $CRT $IMM > bundle.crt

    if [ "`openssl verify bundle.crt`" == "$CRT: OK" ]; then
        echo "Done (bundle ok)"
        exit 0
    fi
fi

while true; do

    if [ "`openssl verify $CRT`" == "$CRT: OK" ]; then
        echo "Done"
        exit 0
    fi

    NEXT=`openssl x509 -noout -issuer_hash -in $CRT`

    if [ ! -f $NEXT ]; then
        echo >&2 "Could not generate trusted bundle"
        exit 1
    fi

    cat $CRT $NEXT > tmp.crt
    mv tmp.crt bundle.crt
    CRT="bundle.crt"

done
