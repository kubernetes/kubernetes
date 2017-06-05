#!/bin/bash

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

# reconfigure docker network setting

source "$HOME/kube/${KUBE_CONFIG_FILE##*/}"

if [[ -n "$DEBUG" ]] && [[ "$DEBUG" != false ]] && [[ "$DEBUG" != FALSE ]]; then
	set -x
fi

if [[ "$(id -u)" != "0" ]]; then
  echo >&2 "Please run as root"
  exit 1
fi

function config_etcd {
  attempt=0
  while true; do
    /opt/bin/etcdctl get /coreos.com/network/config
    if [[ "$?" == 0 ]]; then
      break
    else
    	# enough timeout??
      if (( attempt > 600 )); then
        echo "timeout waiting for /coreos.com/network/config" >> ~/kube/err.log
        exit 2
      fi

      /opt/bin/etcdctl mk /coreos.com/network/config "{\"Network\":\"${FLANNEL_NET}\", \"Backend\": ${FLANNEL_BACKEND:-"{\"Type\": \"vxlan\"}"}${FLANNEL_OTHER_NET_CONFIG}}"
      attempt=$((attempt+1))
      sleep 3
    fi
  done
}

function restart_docker {
  attempt=0
  while [[ ! -f /run/flannel/subnet.env ]]; do 
    if (( attempt > 200 )); then
      echo "timeout waiting for /run/flannel/subnet.env" >> ~/kube/err.log 
      exit 2
    fi
    attempt=$((attempt+1))
    sleep 3
  done
  
  sudo ip link set dev docker0 down
  sudo brctl delbr docker0

  source /run/flannel/subnet.env
  source /etc/default/docker
  echo DOCKER_OPTS=\" -H tcp://127.0.0.1:4243 -H unix:///var/run/docker.sock \
       --bip=${FLANNEL_SUBNET} --mtu=${FLANNEL_MTU}\" > /etc/default/docker
  sudo service docker restart
}

if [[ $1 == "i" ]]; then
  restart_docker
elif [[ $1 == "ai" ]]; then
  config_etcd
  restart_docker
elif [[ $1 == "a" ]]; then
  config_etcd
else
  echo "Another argument is required."
  exit 1
fi 
