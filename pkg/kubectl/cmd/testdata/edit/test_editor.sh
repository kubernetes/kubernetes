#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# Send the file content to the server
curl -s -k -XPOST "${KUBE_EDITOR_CALLBACK}" --data-binary "@${1}" -H "Accept: application/json" -o "${1}.result"
# Use the response as the edited version
mv "${1}.result" "${1}"
