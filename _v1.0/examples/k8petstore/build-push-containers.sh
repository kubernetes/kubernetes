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

#K8PetStore version is tied to the redis version.  We will add more info to version tag later.
#Change the 'jayunit100' string below to you're own dockerhub name and run this script.
#It will build all the containers for this application and publish them to your dockerhub account
version="r.2.8.19"
docker build -t jayunit100/k8-petstore-redis:$version ./redis/
docker build -t jayunit100/k8-petstore-redis-master:$version ./redis-master
docker build -t jayunit100/k8-petstore-redis-slave:$version ./redis-slave
docker build -t jayunit100/k8-petstore-web-server:$version ./web-server

docker push jayunit100/k8-petstore-redis:$version
docker push jayunit100/k8-petstore-redis-master:$version
docker push jayunit100/k8-petstore-redis-slave:$version
docker push jayunit100/k8-petstore-web-server:$version
