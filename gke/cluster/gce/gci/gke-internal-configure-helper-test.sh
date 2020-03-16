#!/bin/bash

set -euo pipefail

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  cd "${RUNFILES_DIR}/${TEST_WORKSPACE}/gke/cluster/gce/gci"
else
  cd "$(dirname $0)"
fi

source gke-internal-configure-helper.sh

echo -n "Testing _gke_cni_template..."
_gke_cni_template | jq . > /dev/null
echo "OK"

exit 0
