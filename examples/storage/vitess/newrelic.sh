#!/bin/bash

# Copyright 2017 Google Inc.
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

NEWRELIC_LICENSE_KEY=$1
VTDATAROOT=$2

./newrelic_start_agent.sh $NEWRELIC_LICENSE_KEY
if [ -n "$VTDATAROOT" ]; then
  sudo cp newrelic_start*.sh $VTDATAROOT
fi

mysql_docker_image=`sudo docker ps | awk '$NF~/^k8s_mysql/ {print $1}'`
vttablet_docker_image=`sudo docker ps | awk '$NF~/^k8s_vttablet/ {print $1}'`
vtgate_docker_image=`sudo docker ps | awk '$NF~/^k8s_vtgate/ {print $1}'`
for image in `echo -e "$mysql_docker_image\n$vttablet_docker_image\n$vtgate_docker_image"`; do
  if [ -z "$VTDATAROOT" ]; then
    vtdataroot=`sudo docker inspect -f '{{index .Volumes "/vt/vtdataroot"}}' $image`
    sudo cp newrelic_start*.sh $vtdataroot
  fi
  sudo docker exec $image apt-get update
  sudo docker exec $image apt-get install sudo -y
  sudo docker exec $image apt-get install procps -y
  sudo docker exec $image bash /vt/vtdataroot/newrelic_start_agent.sh $NEWRELIC_LICENSE_KEY
done
