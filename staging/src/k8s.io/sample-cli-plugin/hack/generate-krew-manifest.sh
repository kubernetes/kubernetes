#!/usr/bin/env bash

# Copyright 2019 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
[[ -n "${DEBUG:-}" ]] && set -x

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

out_dirname="out"
archive="$out_dirname/archive.tar.gz"
mkdir -p "$out_dirname"

if ! hash envsubst 2>/dev/null; then
  echo >&2 'This script requires `envsubst` from the gettext package.'
  exit 1
fi

if [[ ! -f "$archive" ]]; then
  echo >&2 'You have to run create-plugin-archive.sh beforehand.'
  exit 1
fi

ARCHIVE_SHASUM=$(sha256sum "$archive" | sed 's/ .*//') envsubst '$ARCHIVE_SHASUM' \
  < "${SCRIPTDIR}/sample-plugin-template.yaml" \
  > "$out_dirname/sample-plugin.yaml"


echo 'You can now do a test installation of the sample plugin via Krew.'
echo 'If you do not have Krew installed yet, go to https://krew.dev/#installation'
echo ''
echo '$ kubectl krew install --manifest out/sample-plugin.yaml --archive out/archive.tar.gz'
echo ''
echo 'To clean up, run'
echo '$ kubectl krew remove sample-plugin'
