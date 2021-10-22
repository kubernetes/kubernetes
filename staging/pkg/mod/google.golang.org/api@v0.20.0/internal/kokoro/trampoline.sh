#!/bin/bash

# Copyright 2018 Google LLC.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -eo pipefail
# Always run the cleanup script, regardless of the success of bouncing into
# the container.
function cleanup() {
    chmod +x ${KOKORO_GFILE_DIR}/trampoline_cleanup.sh
    ${KOKORO_GFILE_DIR}/trampoline_cleanup.sh
    echo "cleanup";
}
trap cleanup EXIT
python3 "${KOKORO_GFILE_DIR}/trampoline_v1.py"
