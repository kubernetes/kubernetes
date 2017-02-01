#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# send the original content to the server
curl -s -k -XPOST "http://localhost:8081/callback/in" --data-binary "@${1}"
# allow the user to edit the file
vi "${1}"
# send the resulting content to the server
curl -s -k -XPOST "http://localhost:8081/callback/out" --data-binary "@${1}"
