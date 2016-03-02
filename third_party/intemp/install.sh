#!/usr/bin/env bash

# Install intemp from the internet
# Usage: install.sh [version]
# Alt Usage: curl -o- https://raw.githubusercontent.com/karlkfi/intemp/master/install.sh | bash
# Requires: curl

set -o errexit
set -o nounset
set -o pipefail

prefix="/usr/local/bin"

version=${1:-}
if [ -z "${version}" ]; then
  version=$(curl -s https://api.github.com/repos/karlkfi/intemp/releases/latest | grep 'tag_name' | cut -d\" -f4)
fi

echo "Installing intemp ${version} -> ${prefix}/intemp.sh"
curl -o- "https://raw.githubusercontent.com/karlkfi/intemp/${version}/intemp.sh" > "${prefix}/intemp.sh"
chmod a+x "${prefix}/intemp.sh"
