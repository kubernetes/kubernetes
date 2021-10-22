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

generate() {
  dst="$1"
  wsdl="$2"
  modl="$3"

  pkgs=(types methods)
  if [ -n "$modl" ] ; then
    pkgs+=(mo)
  fi

  for p in "${pkgs[@]}"
  do
    mkdir -p "$dst/$p"
  done

  echo "generating $dst/..."

  bundle exec ruby gen_from_wsdl.rb "$dst" "$wsdl"
  if [ -n "$modl" ] ; then
    bundle exec ruby gen_from_vmodl.rb "$dst" "$wsdl" "$modl"
  fi

  for p in "${pkgs[@]}"
  do
    pushd "$dst/$p" >/dev/null
    goimports -w ./*.go
    go install
    popd >/dev/null
  done
}

# ./sdk/ contains the contents of wsdl.zip from vimbase build 8129541

generate "../vim25" "vim" "./rbvmomi/vmodl.db" # from github.com/vmware/rbvmomi@1cc9f9e
generate "../pbm" "pbm"
# originally generated, then manually pruned as there are several vim25 types that are duplicated.
# generate "../lookup" "lookup" # lookup.wsdl from build 4571810
# originally generated, then manually pruned.
# generate "../ssoadmin" "ssoadmin" # ssoadmin.wsdl from PSC
