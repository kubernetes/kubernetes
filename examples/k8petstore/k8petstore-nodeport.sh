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
#kubectl="$GOPATH/src/k8s.io/kubernetes/cluster/kubectl.sh"

kubectl="kubectl"
VERSION="r.2.8.19"
_SECONDS=1000          # number of seconds to measure throughput.
FE="1"                # amount of Web server
LG="1"                # amount of load generators
SLAVE="1"             # amount of redis slaves
TEST="1"              # 0 = Don't run tests, 1 = Do run tests.
NS="default"          # namespace
NODE_PORT=30291     #nodePort, see fe-s.json

kubectl="${1:-$kubectl}"
VERSION="${2:-$VERSION}"
_SECONDS="${3:-$_SECONDS}"   # number of seconds to measure throughput.
FE="${4:-$FE}"       # amount of Web server
LG="${5:-$LG}"        # amount of load generators
SLAVE="${6:-$SLAVE}"     # amount of redis slaves
TEST="${7:-$TEST}"      # 0 = Don't run tests, 1 = Do run tests.
NS="${8:-$NS}"          # namespace
NODE_PORT="${9:-$NODE_PORT}" #nodePort, see fe-s.json
echo "Running w/ args: kubectl $kubectl version $VERSION sec $_SECONDS fe $FE lg $LG slave $SLAVE test = $TEST, NAMESPACE = $NS, NODE_PORT = $NODE_PORT"
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
      "port": 3000,
      "nodePort": $NODE_PORT
    }],
    "selector": {
      "name": "frontend"
    },
    "type": "NodePort"
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

#Get the IP addresses of all Kubernetes nodes.
function getIP {
  #currently this script is only tested on GCE. The following line may need to be updated if k8s is not running on a cloud platform
  NODES_IP=$($kubectl get nodes -o go-template='{{range .items}}{{range .status.addresses}}{{if or (eq .type "ExternalIP") (eq .type "LegacyHostIP")}}{{.address}}{{print "\n"}}{{end}}{{end}}{{end}}')
  TEST_IP=$($kubectl get nodes -o go-template='{{range (index .items 0).status.addresses}}{{if eq .type "ExternalIP"}}{{.address}}{{end}}{{end}}')
  if [ -z "$TEST_IP" ]; then
    TEST_IP=$($kubectl get nodes -o go-template='{{range (index .items 0).status.addresses}}{{if eq .type "LegacyHostIP"}}{{.address}}{{end}}{{end}}')
  fi
  if [ -z "$NODES_IP" ]; then
    echo "Error: Can't get node's IP!!!"
    exit 1
  else
    printf '\n\n\n%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' =
    echo -e "List of nodes' IP addresses:\n$NODES_IP"
    echo -e "Node IP $TEST_IP will be used in the tests in this script"
    printf '%*s\n\n\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' =
  fi
}

function getNodePort {
NODE_PORT=$($kubectl get services/frontend --namespace=$NS -o go-template='{{(index .spec.ports 0).nodePort}}')

if [ -z "$NODE_PORT" ]; then
        echo "Error: Can't get NodePort of services/frontend!!!"
        exit 1
else
        printf '\n\n\n%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' =
        echo -e "NodePort of services/frontend:\n$NODE_PORT"
        echo -e "WARNING: On cloud platforms like GCE, you may need to add a firewall rule to allow TCP traffic on port $NODE_PORT"
        printf '%*s\n\n\n\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' =
fi
}

function pollfor {
  pass_http=0

  ### Test HTTP Server comes up.
  for i in {1..150};
  do
      ### Just testing that the front end comes up.  Not sure how to test total entries etc... (yet)
      echo "Trying curl frontend:3000 via $TEST_IP:$NODE_PORT, attempt ${i}. Expect a few failures while pulling images... "
      curl --max-time 1 --connect-timeout 3 "$TEST_IP:$NODE_PORT" > result
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
        echo "curl : $TEST_IP:$NODE_PORT , $i of $_SECONDS"
        curr_cnt="`curl --max-time 1  --connect-timeout 3 "$TEST_IP:$NODE_PORT/llen"`"
        ### Write CSV File of # of trials / total transcations.
        echo "$i $curr_cnt" >> result
        echo "total transactions so far : $curr_cnt"
        sleep 1
  done
}

create

getIP

getNodePort

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
