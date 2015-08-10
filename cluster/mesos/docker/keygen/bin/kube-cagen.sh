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

# Generates root certificate authority crt and key.
# Writes to <out_dir> (use docker volume or docker export to retrieve files).
# Params:
#   out_dir  - dir to write crt and key to

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

source "util-ssl.sh"

out_dir="${1:-}"
[ -z "${out_dir}" ] && echo "No out_dir supplied (param 1)" && exit 1

cluster::mesos::docker::create_root_certificate_authority "${out_dir}"
