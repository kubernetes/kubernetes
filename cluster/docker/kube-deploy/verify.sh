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

# Verify if k8s cluster works
# TODO not like other providers, we did not verfiy kubectl binary here
function verify() {
  	local -a required_daemon

	if [[ $1 == "master" ]]; then
		required_daemon=("/hyperkube apiserver" "/hyperkube controller-manager" 
			"/hyperkube scheduler"
			"/hyperkube kubelet" "/hyperkube proxy"
			"docker -d -H unix:///var/run/docker-bootstrap.sock" 
			"/usr/local/bin/etcd" "/usr/bin/docker")
	else
		required_daemon=("/hyperkube kubelet" "/hyperkube proxy" 
			"docker -d -H unix:///var/run/docker-bootstrap.sock" "/usr/bin/docker")
	fi

	local daemon
	local ok

	for daemon in "${required_daemon[@]}"; do
		PID=`pgrep -f "${daemon}"`
		if [[ -z $PID ]]; then
			ok=1
	  		printf "[WARN]: $daemon is not running! \n"        
		fi
	done

  	if [[ -z $ok ]]; then
  		printf "... Everything is OK! \n"
  		printf "\n"
	fi

	printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n"

}

verify $1

