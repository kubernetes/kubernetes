#!/bin/bash

# Copyright (c) 2014 VMware, Inc. All Rights Reserved.
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

set -e

# This script shows for every request in a debug trace how long it took
# and the name of the request body.

function body-name {
  (
    xmllint --shell $1 <<EOS
    setns soapenv=http://schemas.xmlsoap.org/soap/envelope/
    xpath name(//soapenv:Body/*)
EOS
  )  | head -1 | sed 's/.*Object is a string : \(.*\)$/\1/'
}

if [ -n "$1" ]; then
  cd $1
fi

for req in $(find . -name '*.req.xml'); do
  base=$(basename $req .req.xml)
  session=$(echo $base | awk -F'-' "{printf \"%d\", \$1}")
  number=$(echo $base | awk -F'-' "{printf \"%d\", \$2}")
  client_log=$(dirname $req)/${session}-client.log
  took=$(awk "/ ${number} took / { print \$4 }" ${client_log})

  printf "%s %8s: %s\n" ${base} ${took} $(body-name $req)
done
