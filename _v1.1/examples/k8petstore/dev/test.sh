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

## First set up the host VM.  That ensures
## we avoid vagrant race conditions.
set -x 

cd hosts/ 
echo "note: the VM must be running before you try this"
echo "if not already running, cd to hosts and run vagrant up"
vagrant provision
#echo "removing containers"
#vagrant ssh -c "sudo docker rm -f $(docker ps -a -q)"
cd ..

## Now spin up the docker containers
## these will run in the ^ host vm above.

vagrant up

## Finally, curl the length, it should be 3 .

x=`curl localhost:3000/llen`

for i in `seq 1 100` do
    if [ x$x == "x3" ]; then 
       echo " passed $3 "
       exit 0
    else
       echo " FAIL" 
    fi
done

exit 1 # if we get here the test obviously failed.
