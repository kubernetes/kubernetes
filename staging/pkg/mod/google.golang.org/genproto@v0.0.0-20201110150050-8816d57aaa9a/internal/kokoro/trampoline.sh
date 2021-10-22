#!/bin/bash
# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
