#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

token_dir=${TOKEN_DIR:-/var/srv/kubernetes}
token_file="${token_dir}/known_tokens.csv"

create_accounts=($@)

touch "${token_file}"
for account in "${create_accounts[@]}"; do
  if grep ",${account}," "${token_file}" ; then
    continue
  fi
  token=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  echo "${token},${account},${account}" >> "${token_file}"
  echo "${token}" > "${token_dir}/${account}.token"
  echo "Added ${account}"
done
