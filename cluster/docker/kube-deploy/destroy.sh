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

# This script contains useful functions which will be called during deployment
# to clear bootstrap daemon and existing hyperkube containers.
# And it is also used to tear down the whole deployment when calling kube-down. 

# Clean bootstrap containers, and then destory the bootstrap daemon
function clear_old_bootstrap {
    echo "... Bootstrap daemon already started, destroying"
    PID=`ps -eaf | grep 'unix:///var/run/docker-bootstrap.sock' | grep -v grep | awk '{print $2}'`

    if [[ ! -z "$PID" ]]; then
        clear_bootstrap_containers
        # Clean all the bootstrap images
        docker -H unix:///var/run/docker-bootstrap.sock rmi \
        `docker -H unix:///var/run/docker-bootstrap.sock images -q` || true

        # Kill bootstrap daemon
        kill -9 $PID
        
        echo "... Clearing bootstrap dir"

        # Have to warn user some dirs may be left
        rm -rf /var/lib/docker-bootstrap || true; \
        echo "Warning: Some directories can not be deleted, you need to clear them manually"
    fi
}

function clear_bootstrap_containers {
  # Clean the bootstrap containers
  containers=`docker -H unix:///var/run/docker-bootstrap.sock ps -aq`
  if [[ ! -z  "$containers" ]]; then
      # Sometimes cleaning fs fails, leaving those garbage for now
      docker -H unix:///var/run/docker-bootstrap.sock rm -vf $containers || true
  else
      echo "Nothing on bootstrap to clear"
  fi
}

# Clear the old kubelet, kube-proxy 
function clear_old_components() {
  echo "... Clearing old components on the Node"

  # Stop & rm
  containers=`docker ps -a | grep -E "kube_in_docker|k8s-master" | awk '{print $1}'`
  if [[ ! -z  "$containers" ]]; then
      docker rm -vf $containers || true
  else
      echo "Nothing to clear"
  fi

  # Just stop, in case users have their own hyperkube containers
  suspicious=`docker ps | grep -E "/hyperkube kubelet|/hyperkube proxy" | awk '{print $1}'`
  if [[ ! -z  "$suspicious" ]]; then
      docker stop $suspicious
      echo "... ... And stopped some users' redundant kubelet"
      stubborn=`docker ps | grep -E "/hyperkube kubelet|/hyperkube proxy" | awk '{print $1}'`
      if [[ "" !=  "$stubborn" ]]; then
          echo "... ... [WARN]: Found some extra kubelet|proxy running, they may fail the deployment"
      fi
  fi
}

# Just clear all
function clear_all() {
  clear_old_bootstrap
  clear_old_components
}

# Make all the functions in this scripts can be run as parameter
# This is useful in kube-down.sh 
$@
