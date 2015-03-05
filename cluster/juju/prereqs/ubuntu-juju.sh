#!/bin/bash

# Copyright 2014 Canonical LTD. All rights reserved.
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

# If you find any bugs within this script - please file bugs against the
# Kubernetes Juju Charms project - located here: https://github.com/whitmo/bundle-kubernetes

function check_for_ppa(){
    grep -s ^ /etc/apt/sources.list /etc/apt/sources.list.d/* | grep juju
}

function package_status(){
    local pkgstatus=`dpkg-query -W --showformat='${Status}\n' $1|grep "install ok installed"`
    if [ "" == "$pkgstatus" ]; then
        echo "Missing package $1"
        sudo apt-get --force-yes --yes install $1
    fi

}

function gather_installation_reqs(){

    ppa_installed=$(check_for_ppa) || ppa_installed=''
    if [[ -z "$ppa_installed" ]]; then
        echo "... Detected missing dependencies.. running"
        echo "... add-apt-repository ppa:juju/stable"
        sudo add-apt-repository ppa:juju/stable
        sudo apt-get update
    fi

    package_status 'juju'
    package_status 'juju-quickstart'
}

