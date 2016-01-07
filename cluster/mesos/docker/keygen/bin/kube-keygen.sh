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

# Generates apiserver crt and key.
# Requires provided hostname to be resolvable (use docker link).
# Requires root certificate in <in_dir> (use docker volume).
# Writes to <out_dir> (use docker volume or docker export to retrieve files).
# Params:
#   hostname - host name of the Kubernetes API Server to resolve into an IP
#   in_dir   - dir to read root certificate from
#   out_dir  - (Optional) dir to write crt and key to  (default=<in_dir>)

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

source "util-ssl.sh"

hostname="${1:-}"
[ -z "${hostname}" ] && echo "No hostname supplied (param 1)" && exit 1

in_dir="${2:-}"
[ -z "${in_dir}" ] && echo "No in_dir supplied (param 2)" && exit 1

out_dir="${3:-${in_dir}}"

# Certificate generation depends on IP being resolvable from the provided hostname.
apiserver_ip="$(resolveip ${hostname})"
apiservice_ip="10.10.10.1" #TODO(karlkfi): extract config

cluster::mesos::docker::create_apiserver_cert "${in_dir}" "${out_dir}" "${apiserver_ip}" "${apiservice_ip}"
