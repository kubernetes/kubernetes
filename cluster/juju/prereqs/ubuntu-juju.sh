#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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


set -o errexit
set -o nounset
set -o pipefail


function check_for_ppa() {
    local repo="$1"
    grep -qsw $repo /etc/apt/sources.list /etc/apt/sources.list.d/*
}

function package_status() {
    local pkgname=$1
    local pkgstatus
    pkgstatus=$(dpkg-query -W --showformat='${Status}\n' "${pkgname}")
    if [[ "${pkgstatus}" != "install ok installed" ]]; then
        echo "Missing package ${pkgname}"
        sudo apt-get --force-yes --yes install ${pkgname}
    fi
}

function gather_installation_reqs() {
    if ! check_for_ppa "juju"; then
        echo "... Detected missing dependencies.. running"
        echo "... add-apt-repository ppa:juju/stable"
        sudo add-apt-repository -y ppa:juju/stable
        sudo apt-get update
    fi

    package_status 'juju'
    package_status 'charm-tools'
}
