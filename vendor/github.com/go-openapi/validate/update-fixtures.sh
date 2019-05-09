#!/bin/bash 

set -eu -o pipefail
dir=$(git rev-parse --show-toplevel)
scratch=$(mktemp -d -t tmp.XXXXXXXXXX)

function finish {
  rm -rf "$scratch"
}
trap finish EXIT SIGHUP SIGINT SIGTERM

cd "$scratch"
git clone https://github.com/json-schema-org/JSON-Schema-Test-Suite Suite
cp -r Suite/tests/draft4/* "$dir/fixtures/jsonschema_suite"
cp -a Suite/remotes "$dir/fixtures/jsonschema_suite"
