#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

echoOK() {
    TC='\e['
    RegB="${TC}0m"
    if [ "$1" -eq "0" ]; then
        Green="${TC}32m"
        echo -e "[${Green}OK${RegB}]"
    else
        Red="${TC}31m"
        echo -e "[${Red}FAIL${RegB}]"
        echo "Check log file."
        exit 1
    fi
}

usage() {
    echo "Usage options: [--logfile <path to file>]"
}

logfile=/dev/null
while [[ $# > 0 ]]; do
    key="$1"
    shift
    case $key in
       -l|--logfile)
         logfile="$1"
         if [ "$logfile" == "" ]; then
             usage
             exit 1
         fi
         shift
         ;;
       *)
         # unknown option
         usage
         exit 1
         ;;
       esac
done

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
cd "${KUBE_ROOT}"

echo All verbose output will be redirected to $logfile, use --logfile option to change.

printf "Start the cluster with 2 nodes .. "
export NUM_NODES=2
export KUBERNETES_PROVIDER=vagrant

(cluster/kube-up.sh >>"$logfile" 2>&1) || true
echoOK $?

printf "Check if node-1 can reach kubernetes master .. "
vagrant ssh node-1 -- ping -c 10 kubernetes-master >>"$logfile" 2>&1
echoOK $?
printf "Check if node-2 can reach kubernetes master .. "
vagrant ssh node-2 -- ping -c 10 kubernetes-master >>"$logfile" 2>&1
echoOK $?

printf "Pull an image that runs a web server on node-1 .. "
vagrant ssh node-1 -- 'sudo docker pull kubernetes/serve_hostname' >>"$logfile" 2>&1
echoOK $?
printf "Pull an image that runs a web server on node-2 .. "
vagrant ssh node-2 -- 'sudo docker pull kubernetes/serve_hostname' >>"$logfile" 2>&1
echoOK $?

printf "Run the server on node-1 .. "
vagrant ssh node-1 -- sudo docker run -d kubernetes/serve_hostname >>"$logfile" 2>&1
echoOK $?
printf "Run the server on node-2 .. "
vagrant ssh node-2 -- sudo docker run -d kubernetes/serve_hostname >>"$logfile" 2>&1
echoOK $?

printf "Run ping from node-1 to docker bridges and to the containers on both nodes .. "
vagrant ssh node-1 -- 'ping -c 20 10.246.0.1 && ping -c 20 10.246.1.1 && ping -c 20 10.246.0.2 && ping -c 20 10.246.1.2' >>"$logfile" 2>&1
echoOK $?
printf "Same pinch from node-2 .. "
vagrant ssh node-2 -- 'ping -c 20 10.246.0.1 && ping -c 20 10.246.1.1 && ping -c 20 10.246.0.2 && ping -c 20 10.246.1.2' >>"$logfile" 2>&1
echoOK $?

printf "tcp check, curl to both the running webservers from node-1 .. "
vagrant ssh node-1 -- 'curl -sS 10.246.0.2:9376  && curl -sS 10.246.1.2:9376' >>"$logfile" 2>&1
echoOK $?
printf "tcp check, curl to both the running webservers from node-2 .. "
vagrant ssh node-2 -- 'curl -sS 10.246.0.2:9376  && curl -sS 10.246.1.2:9376' >>"$logfile" 2>&1
echoOK $?

printf "All good, destroy the cluster .. "
vagrant destroy -f >>"$logfile" 2>&1
echoOK $?
