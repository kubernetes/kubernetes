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

echo "WRITING KUBE FILES , will overwrite the jsons, then testing pods. is kube clean ready to go?"


#Args below can be overriden when calling from cmd line.
#Just send all the args in order.
#for dev/test you can use:
#kubectl=$GOPATH/src/github.com/kubernetes/kubernetes/cluster/kubectl.sh"
kubectl="kubectl"
VERSION="r.2.8.19"
PUBLIC_IP="10.1.4.89" # ip which we use to access the Web server.
_SECONDS=1000          # number of seconds to measure throughput.
FE="1"                # amount of Web server  
LG="1"                # amount of load generators
SLAVE="1"             # amount of redis slaves 
TEST="1"              # 0 = Don't run tests, 1 = Do run tests.
NS="default"       # namespace

kubectl="${1:-$kubectl}"
VERSION="${2:-$VERSION}"
PUBLIC_IP="${3:-$PUBLIC_IP}" # ip which we use to access the Web server.
_SECONDS="${4:-$_SECONDS}"   # number of seconds to measure throughput.
FE="${5:-$FE}"       # amount of Web server  
LG="${6:-$LG}"        # amount of load generators
SLAVE="${7:-$SLAVE}"     # amount of redis slaves 
TEST="${8:-$TEST}"      # 0 = Don't run tests, 1 = Do run tests.
NS="${9:-$NS}"          # namespace

echo "Running w/ args: kubectl $kubectl version $VERSION ip $PUBLIC_IP sec $_SECONDS fe $FE lg $LG slave $SLAVE test $TEST NAMESPACE $NS"
function create { 

cat << EOF > fe-rc.json
{
  "kind": "ReplicationController",
  "apiVersion": "v1",
  "metadata": {
    "name": "fectrl",
    "labels": {"name": "frontend"}
  },
  "spec": {
    "replicas": $FE,
    "selector": {"name": "frontend"},
    "template": {
      "metadata": {
        "labels": {
          "name": "frontend",
          "uses": "redis-master"
        }
      },
      "spec": {
         "containers": [{
           "name": "frontend-go-restapi",
           "image": "jayunit100/k8-petstore-web-server:$VERSION"
         }]
      }
    }
  }
}
EOF

cat << EOF > bps-load-gen-rc.json
{
  "kind": "ReplicationController",
  "apiVersion": "v1",
  "metadata": {
    "name": "bpsloadgenrc",
    "labels": {"name": "bpsLoadGenController"}
  },
  "spec": {
    "replicas": $LG,
    "selector": {"name": "bps"},
    "template": {
      "metadata": {
        "labels": {
          "name": "bps",
          "uses": "frontend"
        }
      },
      "spec": {
        "containers": [{
           "name": "bps",
           "image": "jayunit100/bigpetstore-load-generator",
           "command": ["sh","-c","/opt/PetStoreLoadGenerator-1.0/bin/PetStoreLoadGenerator http://\$FRONTEND_SERVICE_HOST:3000/rpush/k8petstore/ 4 4 1000 123"]
        }]
      }
    }
  }
}
EOF

cat << EOF > fe-s.json
{
  "kind": "Service",
  "apiVersion": "v1",
  "metadata": {
    "name": "frontend",
    "labels": {
      "name": "frontend"
    }
  },
  "spec": {
    "ports": [{
      "port": 3000
    }],
    "deprecatedPublicIPs":["$PUBLIC_IP","10.1.4.89"],
    "selector": {
      "name": "frontend"
    }
  }
}
EOF

cat << EOF > rm.json
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "redismaster",
    "labels": {
      "name": "redis-master"
    }
  },
  "spec": {
    "containers": [{
      "name": "master",
      "image": "jayunit100/k8-petstore-redis-master:$VERSION",
      "ports": [{
        "containerPort": 6379
      }]
    }]
  }
}
EOF

cat << EOF > rm-s.json
{
  "kind": "Service",
  "apiVersion": "v1",
  "metadata": {
    "name": "redismaster",
    "labels": {
      "name": "redis-master"
    }
  },
  "spec": {
    "ports": [{
      "port": 6379
    }],
    "selector": {
      "name": "redis-master"
    }
  }
}
EOF

cat << EOF > rs-s.json
{
  "kind": "Service",
  "apiVersion": "v1",
  "metadata": {
    "name": "redisslave",
    "labels": {
      "name": "redisslave"
    }
  },
  "spec": {
    "ports": [{
      "port": 6379
    }],
    "selector": {
      "name": "redisslave"
    }
  }
}
EOF

cat << EOF > slave-rc.json
{
  "kind": "ReplicationController",
  "apiVersion": "v1",
  "metadata": {
    "name": "redissc",
    "labels": {"name": "redisslave"}
  },
  "spec": {
    "replicas": $SLAVE,
    "selector": {"name": "redisslave"},
    "template": {
      "metadata": {
        "labels": {
          "name": "redisslave",
          "uses": "redis-master"
        }
      },
      "spec": {
         "containers": [{
           "name": "slave",
           "image": "jayunit100/k8-petstore-redis-slave:$VERSION",
           "ports": [{"containerPort": 6379}]
         }]
      }
    }
  }
}
EOF
$kubectl create -f rm.json --namespace=$NS
$kubectl create -f rm-s.json --namespace=$NS
sleep 3 # precaution to prevent fe from spinning up too soon.
$kubectl create -f slave-rc.json --namespace=$NS
$kubectl create -f rs-s.json --namespace=$NS
sleep 3 # see above comment.
$kubectl create -f fe-rc.json --namespace=$NS
$kubectl create -f fe-s.json --namespace=$NS
$kubectl create -f bps-load-gen-rc.json --namespace=$NS
}

function pollfor {
  pass_http=0

  ### Test HTTP Server comes up.
  for i in {1..150};
  do
      ### Just testing that the front end comes up.  Not sure how to test total entries etc... (yet)
      echo "Trying curl ... $PUBLIC_IP:3000 , attempt $i . expect a few failures while pulling images... "
      curl --max-time 1 "$PUBLIC_IP:3000" > result
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
    for i in `seq 1 $_SECONDS`;
     do
        echo "curl : $PUBLIC_IP:3000 , $i of $_SECONDS"
        curr_cnt="`curl --max-time 1 "$PUBLIC_IP:3000/llen"`" 
        ### Write CSV File of # of trials / total transcations.
        echo "$i $curr_cnt" >> result
        echo "total transactions so far : $curr_cnt"
        sleep 1
  done
}

function warning {
  echo ""
  echo "THIS SCRIPT IS FOR KUBERNETES < v1."
  echo "For LATER VERSIONS, use k8petstore-nodeport.sh or k8petstore-loadbalacer.sh!!!!"
  echo "In particular PublicIP is DEPRECATED in post-v1 releases!!!"
  echo ""
}

warning

create

pollfor

if [[ $pass_http -eq 1 ]]; then 
    echo "Passed..."
else
    exit 1
fi

if [[ $TEST -eq 1 ]]; then
    echo "running polling tests now"
    tests
fi
