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

#!/bin/bash

source ../../hack/lib/util.sh  # need kube::util::host_platform()

HOST_OS=$(kube::util::host_platform)
HOST_OS=${HOST_OS%/*}  # just host_os name

function setup_vm() {
    ### Provider = vbox.  You can use another one if you want... But untested.
    PROVIDER=virtualbox

    ### Create a VM specific to this app...
    if docker-machine ls | grep -q k8petstore ; then 
      echo "VM already exists, moving on..."
    else
      docker-machine create --driver $PROVIDER k8petstore
    fi
}

function setup_docker() {

    ## Set the docker server, and then clean all containers... 
    eval "$(docker-machine env k8petstore)"
    docker rm -f `docker ps -a -q`

}

function build_containers() {

    version="`date +"%m-%d-%Y-%s"`"
    pushd redis
    docker build -t jayunit100/k8-petstore-redis:$version ./
    popd

    pushd redis-master
    docker build -t jayunit100/k8-petstore-redis-master:$version ./    
    popd
    
    pushd redis-slave
    docker build -t jayunit100/k8-petstore-redis-slave:$version ./
    popd
    
    pushd web-server
    docker build -t jayunit100/k8-petstore-web-server:$version ./
    popd
}

function runk8petstore() {

    ### Finally, run the application.
    ### This app is guaranteed to be a clean run using all the source.
    ### You can use it to iteratively test/deploy k8petstore and make new changes.

    ### TODO, add slaves.

    echo "Running k8petstore now..."
    docker run --name redis -d -p 6379:6379 jayunit100/k8-petstore-redis-master:$version
    docker run --link redis:redis -d -e REDISMASTER_SERVICE_HOST=redis -e REDISMASTER_SERVICE_PORT=6379 -p 3000:3000 jayunit100/k8-petstore-web-server:$version
    
}


if [[ "$HOST_OS" != linux ]] ; then
  setup_vm
  setup_docker
fi

build_containers

runk8petstore
