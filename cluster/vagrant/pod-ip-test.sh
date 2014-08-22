#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

cd $(dirname ${BASH_SOURCE})/../../

echo All verbose output will be redirected to $logfile, use --logfile option to change.

printf "Start the cluster with 2 minions .. "
export KUBERNETES_NUM_MINIONS=2
export KUBERNETES_PROVIDER=vagrant

(cluster/kube-up.sh &>> $logfile) || true
echoOK $?

printf "Check if minion-1 can reach kubernetes master .. "
vagrant ssh minion-1 -- ping -c 10 kubernetes-master &>> $logfile
echoOK $?
printf "Check if minion-2 can reach kubernetes master .. "
vagrant ssh minion-2 -- ping -c 10 kubernetes-master &>> $logfile
echoOK $?

printf "Pull an image that runs a web server on minion-1 .. "
vagrant ssh minion-1 -- 'sudo docker pull dockerfile/nginx' &>> $logfile
echoOK $?
printf "Pull an image that runs a web server on minion-2 .. "
vagrant ssh minion-2 -- 'sudo docker pull dockerfile/nginx' &>> $logfile
echoOK $?

printf "Run the server on minion-1 .. "
vagrant ssh minion-1 -- sudo docker run -d dockerfile/nginx &>> $logfile
echoOK $?
printf "Run the server on minion-2 .. "
vagrant ssh minion-2 -- sudo docker run -d dockerfile/nginx &>> $logfile
echoOK $?

printf "Run ping from minion-1 to docker bridges and to the containers on both minions .. "
vagrant ssh minion-1 -- 'ping -c 20 10.244.1.1 && ping -c 20 10.244.2.1 && ping -c 20 10.244.1.3 && ping -c 20 10.244.2.3' &>> $logfile
echoOK $?
printf "Same pinch from minion-2 .. "
vagrant ssh minion-2 -- 'ping -c 20 10.244.1.1 && ping -c 20 10.244.2.1 && ping -c 20 10.244.1.3 && ping -c 20 10.244.2.3' &>> $logfile
echoOK $?

printf "tcp check, curl to both the running webservers from minion-1 .. "
vagrant ssh minion-1 -- 'curl 10.244.1.3:80  && curl 10.244.2.3:80' &>> $logfile
echoOK $?
printf "tcp check, curl to both the running webservers from minion-2 .. "
vagrant ssh minion-2 -- 'curl 10.244.1.3:80 && curl 10.244.2.3:80' &>> $logfile
echoOK $?

printf "All good, destroy the cluster .. "
vagrant destroy -f &>> $logfile
echoOK $?
