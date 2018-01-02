#!/bin/bash
set -e -x

cd $(dirname $0)/../generator

URL_BASE='http://localhost:8080'

if [ "$1" != "" ]; then
    URL_BASE=$1
fi

echo -n Waiting for cattle ${URL_BASE}/ping
while ! curl -fs ${URL_BASE}/ping; do
    echo -n .
    sleep 1
done
echo

gen() {
    BASE=$1

    curl -s "${URL_BASE}/$BASE/schemas?_role=service" | jq . > schemas.json
    echo Saved schemas.json

    echo -n Generating go code...
    rm -rf ../client/generated_* || true
    go run generator.go
    echo " Done"

    gofmt -w ../client/generated_*
    echo Formatted code

    if [ -n "$2" ]; then
        rm -rf ../$2
        mv ../client ../$2
        if [ -n "$3" ]; then
            sed -i -e 's/package client/package '$2'/g' ../$2/*.go
        fi
        git checkout ../client
    fi
    rm schemas.json
}

gen v1-catalog catalog rename
gen v2-beta v2
gen v1

echo Success
