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


# get the full path of configure dir and set $PWD to it.
CONFIG_DIR=`dirname "$0"`
CONFIG_DIR=`cd "$CONFIG_DIR"; pwd`
cd $CONFIG_DIR

#clean all init/init.d/configs
function do_backup_clean() {
  #backup all config files
  init_files=`ls init_conf`
  for i in $init_files
  do
    if [ -f /etc/init/$i ]
    then
      mv /etc/init/${i} /etc/init/${i}.bak
    fi
  done
  initd_files=`ls initd_scripts`
  for i in $initd_files
  do
    if [ -f /etc/init.d/$i ]
    then
      mv /etc/init.d/${i} /etc/init.d/${i}.bak
    fi
  done
  default_files=`ls default_scripts`
  for i in $default_files
  do
    if [ -e /etc/default/$i ]
    then
      mv /etc/default/${i} /etc/default/${i}.bak
    fi
  done
  # clean work dir 
  if [ ! -d ./work ]
  then
    mkdir work
  fi
    cp -rf default_scripts init_conf initd_scripts work
}

function cpMaster(){
	# copy /etc/init files
    cp work/init_conf/etcd.conf /etc/init/
    cp work/init_conf/kube-apiserver.conf /etc/init/
    cp work/init_conf/kube-controller-manager.conf /etc/init/
    cp work/init_conf/kube-scheduler.conf /etc/init/

    # copy /etc/initd/ files
    cp work/initd_scripts/etcd /etc/init.d/
    cp work/initd_scripts/kube-apiserver /etc/init.d/
    cp work/initd_scripts/kube-controller-manager /etc/init.d/
    cp work/initd_scripts/kube-scheduler /etc/init.d/

    # copy default configs
    cp work/default_scripts/etcd /etc/default/
    cp work/default_scripts/kube-apiserver /etc/default/
    cp work/default_scripts/kube-scheduler /etc/default/
    cp work/default_scripts/kube-controller-manager /etc/default/
}

function cpMinion(){
	# copy /etc/init files
    cp work/init_conf/etcd.conf /etc/init/etcd.conf
    cp work/init_conf/kubelet.conf /etc/init/kubelet.conf
    cp work/init_conf/flanneld.conf /etc/init/flanneld.conf
    cp work/init_conf/kube-proxy.conf /etc/init/

    # copy /etc/initd/ files
    cp work/initd_scripts/etcd /etc/init.d/
    cp work/initd_scripts/flanneld /etc/init.d/
    cp work/initd_scripts/kubelet /etc/init.d/
    cp work/initd_scripts/kube-proxy /etc/init.d/

    # copy default configs
    cp work/default_scripts/etcd /etc/default/
    cp work/default_scripts/flanneld /etc/default/
    cp work/default_scripts/kube-proxy /etc/default/
    cp work/default_scripts/kubelet /etc/default/
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
    echo ETCD_OPTS=\"-name $1 -initial-advertise-peer-urls http://$2:2380 -listen-peer-urls http://$2:2380 -initial-cluster-token etcd-cluster-1 -initial-cluster $3 -initial-cluster-state new\" > work/default_scripts/etcd	
}

# check root
if [ "$(id -u)" != "0" ]; then
    echo >&2 "Please run as root"
    exit 1
fi

echo "Welcome to use this script to configure k8s setup"

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

# check kube commands
if ! $(which kube-apiserver > /dev/null) && ! $(which kubelet > /dev/null)
then
    echo "warning: kube binaries are not found in the $PATH"
    exit 1
fi

# detect the etcd version, we support only etcd 2.0.
etcdVersion=$(/opt/bin/etcd --version | awk '{print $3}')

if [ "$etcdVersion" != "2.0.0" ]; then
	echo "We only support 2.0.0 version of etcd"
	exit 1
fi

do_backup_clean

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
            # For minion set MINION IP in default_scripts/kubelet
            sed -i "s/MY_IP/${myIP}/g" work/default_scripts/kubelet
            sed -i "s/MASTER_IP/${masterIP}/g" work/default_scripts/kubelet
            sed -i "s/MASTER_IP/${masterIP}/g" work/default_scripts/kube-proxy
            
            # For master set MINION IPs in kube-controller-manager
	    if [ -z "$minionIPs" ]; then
                #one node act as both minion and master role
                minionIPs="$myIP"
            else
                minionIPs="$minionIPs,$myIP"
            fi

	        sed -i "s/MINION_IPS/${minionIPs}/g" work/default_scripts/kube-controller-manager
	        
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
            # set MINION IPs in kube-controller-manager
            sed -i "s/MINION_IPS/${minionIPs}/g" work/default_scripts/kube-controller-manager
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
            # set MINION IP in default_scripts/kubelet
            sed -i "s/MY_IP/${myIP}/g" work/default_scripts/kubelet
            sed -i "s/MASTER_IP/${masterIP}/g" work/default_scripts/kubelet
            sed -i "s/MASTER_IP/${masterIP}/g" work/default_scripts/kube-proxy
	        cpMinion
	        break
	        ;;
	    * )
	        echo "Please choose 1 or 2 or 3."
	        ;;
	esac
done

echo -e "\e[0;32mConfigure Success\033[0m"
