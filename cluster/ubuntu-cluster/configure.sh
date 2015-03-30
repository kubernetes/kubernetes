#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# simple use the sed to replace some ip settings on user's demand
# Run as root only

# author @WIZARD-CXY @resouer

set -e

function cpMaster(){
	# copy /etc/init files
    cp init_conf/etcd.conf /etc/init/
    cp init_conf/lmktfy-apiserver.conf /etc/init/
    cp init_conf/lmktfy-controller-manager.conf /etc/init/
    cp init_conf/lmktfy-scheduler.conf /etc/init/

    # copy /etc/initd/ files
    cp initd_scripts/etcd /etc/init.d/
    cp initd_scripts/lmktfy-apiserver /etc/init.d/
    cp initd_scripts/lmktfy-controller-manager /etc/init.d/
    cp initd_scripts/lmktfy-scheduler /etc/init.d/

    # copy default configs
    cp default_scripts/etcd /etc/default/
    cp default_scripts/lmktfy-apiserver /etc/default/
    cp default_scripts/lmktfy-scheduler /etc/default/
    cp default_scripts/lmktfy-controller-manager /etc/default/
}

function cpMinion(){
	# copy /etc/init files
    cp init_conf/etcd.conf /etc/init/
    cp init_conf/lmktfylet.conf /etc/init/
    cp init_conf/flanneld.conf /etc/init/
    cp init_conf/lmktfy-proxy.conf /etc/init/

    # copy /etc/initd/ files
    cp initd_scripts/etcd /etc/init.d/
    cp initd_scripts/flanneld /etc/init.d/
    cp initd_scripts/lmktfylet /etc/init.d/
    cp initd_scripts/lmktfy-proxy /etc/init.d/

    # copy default configs
    cp default_scripts/etcd /etc/default/
    cp default_scripts/flanneld /etc/default/
    cp default_scripts/lmktfy-proxy /etc/default
    cp default_scripts/lmktfylet /etc/default/
}

# check if input IP in machine list
function inList(){
	if [ "$#" -eq 1 ]; then
		echo -e "\e[0;31mERROR\e[0m: "$1" is not in your machine list."
		exit 1
	fi
}

# set values in ETCD_OPTS
function configEtcd(){
    echo ETCD_OPTS=\"-name $1 -initial-advertise-peer-urls http://$2:2380 -listen-peer-urls http://$2:2380 -initial-cluster-token etcd-cluster-1 -initial-cluster $3 -initial-cluster-state new\" > default_scripts/etcd	
}

# check root
if [ "$(id -u)" != "0" ]; then
    echo >&2 "Please run as root"
    exit 1
fi

echo "Welcome to use this script to configure lmktfy setup"

echo

PATH=$PATH:/opt/bin

# use ubuntu
if ! $(grep Ubuntu /etc/lsb-release > /dev/null 2>&1)
then
    echo "warning: not detecting a ubuntu system"
    exit 1
fi

# check etcd
if ! $(which etcd > /dev/null)
then
    echo "warning: etcd binary is not found in the PATH: $PATH"
    exit 1
fi

# check lmktfy commands
if ! $(which lmktfy-apiserver > /dev/null) && ! $(which lmktfylet > /dev/null)
then
    echo "warning: lmktfy binaries are not found in the $PATH"
    exit 1
fi

# detect the etcd version, we support only etcd 2.0.
etcdVersion=$(/opt/bin/etcd --version | awk '{print $3}')

if [ "$etcdVersion" != "2.0.0" ]; then
	echo "We only support 2.0.0 version of etcd"
	exit 1
fi


# use an array to record name and ip
declare -A mm
ii=1
# we use static etcd configuration 
# see https://github.com/coreos/etcd/blob/master/Documentation/clustering.md#static
echo "Please enter all your cluster node ips, MASTER node comes first"
read -p "And separated with blank space like \"<ip_1> <ip_2> <ip_3>\": " etcdIPs

for i in $etcdIPs
do
    name="infra"$ii
    item="$name=http://$i:2380"
    if [ "$ii" == 1 ]; then 
        cluster=$item
        #record the masterIP for later use.
        masterIP=$i
    else
        cluster="$cluster,$item"
        if [ "$ii" -gt 2 ]; then
        	    minionIPs="$minionIPs,$i"
        else
        	    minionIPs="$i"
        fi
    fi
    mm[$i]=$name
    let ii++
done

# input node IPs
while true; do
    echo "This machine acts as"
    echo -e "  both MASTER and MINION:      \033[1m1\033[0m"
    echo -e "  only MASTER:                 \033[1m2\033[0m"
    echo -e "  only MINION:                 \033[1m3\033[0m"
	read -p "Please choose a role > " option 
    echo

	case $option in
	    [1] )
            # as both master and minion
        	read -p "IP address of this machine > " myIP
            echo
            etcdName=${mm[$myIP]}
            inList $etcdName $myIP
            configEtcd $etcdName $myIP $cluster
            # For minion set MINION IP in default_scripts/lmktfylet
            sed -i "s/MY_IP/${myIP}/g" default_scripts/lmktfylet
            sed -i "s/MASTER_IP/${masterIP}/g" default_scripts/lmktfylet         
            sed -i "s/MASTER_IP/${masterIP}/g" default_scripts/lmktfy-proxy        
            
            # For master set MINION IPs in lmktfy-controller-manager
            if [ -z "$minionIPs" ]; then
                #one node act as both minion and master role
                minionIPs="$myIP"
            else
                minionIPs="$minionIPs,$myIP"
            fi

	        sed -i "s/MINION_IPS/${minionIPs}/g" default_scripts/lmktfy-controller-manager
	        
	        cpMaster
	        cpMinion
	        break
	        ;;
        [2] )
            # as master
        	read -p "IP address of this machine > " myIP
            echo
            etcdName=${mm[$myIP]}
            inList $etcdName $myIP
            configEtcd $etcdName $myIP $cluster
            # set MINION IPs in lmktfy-controller-manager
            sed -i "s/MINION_IPS/${minionIPs}/g" default_scripts/lmktfy-controller-manager
	        cpMaster
	        break
            ;;
        [3] )
            # as minion
        	read -p "IP address of this machine > " myIP
            echo
            etcdName=${mm[$myIP]}
            inList $etcdName $myIP
            configEtcd $etcdName $myIP $cluster
            # set MINION IP in default_scripts/lmktfylet
            sed -i "s/MY_IP/${myIP}/g" default_scripts/lmktfylet
            sed -i "s/MASTER_IP/${masterIP}/g" default_scripts/lmktfylet
	        cpMinion
	        break
	        ;;
	    * )
	        echo "Please choose 1 or 2 or 3."
	        ;;
	esac
done

echo -e "\e[0;32mConfigure Success\033[0m"
