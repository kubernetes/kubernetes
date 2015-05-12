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

echo "WRITING KUBE FILES , will overwrite the jsons, then testing pods. is kube clean ready to go?"


#Args below can be overriden when calling from cmd line.
#Just send all the args in order.
#for dev/test you can use:
#kubectl=$GOPATH/src/github.com/GoogleCloudPlatform/kubernetes/cluster/kubectl.sh"
kubectl="kubectl"
VERSION="r.2.8.19"
PUBLIC_IP="10.1.4.89" # ip which we use to access the Web server.
FE="1"                # amount of Web server  
LG="1"                # amount of load generators
SLAVE="1"             # amount of redis slaves 
TEST_SECONDS="1000"   # 0 = Dont run tests, if > 0,  run tests for n seconds.
NS="k8petstore"       # namespace

kubectl="${1:-$kubectl}"
VERSION="${2:-$VERSION}"
PUBLIC_IP="${3:-$PUBLIC_IP}"
FE="${4:-$FE}"  
LG="${5:-$LG}"
SLAVE="${6:-$SLAVE}" 
TEST_SECONDS="${7:-$TEST}"
NS="${8:-$NS}" 

echo "Running w/ args: kubectl $kubectl version $VERSION ip $PUBLIC_IP sec $_SECONDS fe $FE lg $LG slave $SLAVE test $TEST NAMESPACE $NS"
function create { 

cat << EOF > fe-rc.json
{
  "id": "fectrl",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": $FE,
    "replicaSelector": {"name": "frontend"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "frontendCcontroller",
           "containers": [{
             "name": "frontend-go-restapi",
             "image": "jayunit100/k8-petstore-web-server:$VERSION"
         }]
         }
       },
       "labels": {
         "name": "frontend",
         "uses": "redis-master"
       }
      }},
  "labels": {"name": "frontend"}
}
EOF

cat << EOF > bps-load-gen-rc.json
{
  "id": "bpsloadgenrc",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": $LG,
    "replicaSelector": {"name": "bps"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "bpsLoadGenController",
           "containers": [{
             "name": "bps",
             "image": "jayunit100/bigpetstore-load-generator",
             "command": ["sh","-c","/opt/PetStoreLoadGenerator-1.0/bin/PetStoreLoadGenerator http://\$FRONTEND_SERVICE_HOST:3000/rpush/k8petstore/ 4 4 1000 123"]
         }]
         }
       },
       "labels": {
         "name": "bps",
         "uses": "frontend"
        }
      }},
  "labels": {"name": "bpsLoadGenController"}
}
EOF

cat << EOF > fe-s.json
{
  "id": "frontend",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 3000,
  "containerPort": 3000,
  "publicIPs":["$PUBLIC_IP","10.1.4.89"],
  "selector": {
    "name": "frontend"
  },
  "labels": {
    "name": "frontend"
  }
}
EOF

cat << EOF > rm.json
{
  "id": "redismaster",
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "redismaster",
      "containers": [{
        "name": "master",
        "image": "jayunit100/k8-petstore-redis-master:$VERSION",
        "ports": [{
          "containerPort": 6379,
          "hostPort": 6379
        }]
      }]
    }
  },
  "labels": {
    "name": "redis-master"
  }
}
EOF

cat << EOF > rm-s.json
{
  "id": "redismaster",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 6379,
  "containerPort": 6379,
  "selector": {
    "name": "redis-master"
  },
  "labels": {
    "name": "redis-master"
  }
}
EOF

cat << EOF > rs-s.json
{
  "id": "redisslave",
  "kind": "Service",
  "apiVersion": "v1beta1",
  "port": 6379,
  "containerPort": 6379,
  "labels": {
    "name": "redisslave"
  },
  "selector": {
    "name": "redisslave"
  }
}
EOF

cat << EOF > slave-rc.json
{
  "id": "redissc",
  "kind": "ReplicationController",
  "apiVersion": "v1beta1",
  "desiredState": {
    "replicas": $SLAVE,
    "replicaSelector": {"name": "redisslave"},
    "podTemplate": {
      "desiredState": {
         "manifest": {
           "version": "v1beta1",
           "id": "redissc",
           "containers": [{
             "name": "slave",
             "image": "jayunit100/k8-petstore-redis-slave:$VERSION",
             "ports": [{"containerPort": 6379, "hostPort": 6380}]
           }]
         }
      },
      "labels": {
        "name": "redisslave",
        "uses": "redis-master"
      }
    }
  },
  "labels": {"name": "redisslave"}
}
EOF
$kubectl create -f rm.json --api-version=v1beta1 --namespace=$NS
$kubectl create -f rm-s.json --api-version=v1beta1 --namespace=$NS
sleep 3 # precaution to prevent fe from spinning up too soon.
$kubectl create -f slave-rc.json --api-version=v1beta1 --namespace=$NS
$kubectl create -f rs-s.json --api-version=v1beta1 --namespace=$NS
sleep 3 # see above comment.
$kubectl create -f fe-rc.json --api-version=v1beta1 --namespace=$NS
$kubectl create -f fe-s.json --api-version=v1beta1 --namespace=$NS
$kubectl create -f bps-load-gen-rc.json --api-version=v1beta1 --namespace=$NS
}

function pollfor {
	pass_http=0

	### Test HTTP Server comes up.
	for i in `seq 1 150`;
	do
	    ### Just testing that the front end comes up.  Not sure how to test total entries etc... (yet)
	    echo "Trying curl ... $PUBLIC_IP:3000 , attempt $i . expect a few failures while pulling images... " 
	    curl "$PUBLIC_IP:3000" > result
	    cat result
	    cat result | grep -q "k8-bps"
	    if [ $? -eq 0 ]; then
    		echo "TEST PASSED after $i tries !"
	    	i=1000
		    break
	    else	
		echo "the above RESULT didn't contain target string for trial $i"
	    fi
	    sleep 3
	done

	if [ $i -eq 1000 ]; then
	   pass_http=1
	fi
    
}

function tests {
	pass_load=0 

	### Print statistics of db size, every second, until $SECONDS are up.
	for i in `seq 1 $TEST_SECONDS`;
	 do
	    echo "curl : $PUBLIC_IP:3000 , $i of $TEST_SECONDS"
        curr_cnt="`curl "$PUBLIC_IP:3000/llen"`" 
        ### Write CSV File of # of trials / total transcations.
        echo "$i $curr_cnt" >> result
        echo "total transactions so far : $curr_cnt"
	    sleep 1
	done
}

create

pollfor

if [[ $pass_http -eq 1 ]]; then 
    echo "Passed..."
else
    exit 2
fi

if [[ $TEST_SECONDS -eq 0 ]]; then
    echo "skipping tests, TEST_SECONDS value was 0"
else
    echo "running polling tests now for $TEST_SECONDS"
    tests
fi

exit 0
