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

set -o errexit
set -o nounset
set -o pipefail
set -x

# Grep the whole IP because otherwise sometimes vagrant attaches extra mystery dynamic IPs to eth1.
IP=`ip -o addr | grep '192.168.4' | cut -d' ' -f 7 | cut -d'/' -f 1`

echo "Using IP $IP for this machine."

function initialize {
    systemctl disable iptables-services firewalld
    echo "disabling selinux"
    (setenforce 0 || echo "selinux might already be disabled...")
    yum install -y docker

    ### Important : The kube version MUST match the containers in the manifests.
    ### Otherwise lots of API errors.
    yum install -y http://cbs.centos.org/kojifiles/packages/kubernetes/0.17.1/3.el7/x86_64/kubernetes-node-0.17.1-3.el7.x86_64.rpm
    mkdir -p -m 777 /etc/kubernetes/manifests
    ### just to make it easy to hack around as non root user
    groupadd docker
    gpasswd -a vagrant docker
    systemctl restart docker
}

function start_kubelet {
    systemctl enable docker
    ### We need a custom unit file with the --config/ option
    cp /vagrant/etc_kubernetes_kubelet /etc/kubernetes/kubelet
    systemctl enable kubelet
    ### Not sure why, but this restart is required?
    sleep 2
    systemctl restart kubelet
}

### Not the best idea if using flannel.  Because of the circular dependency.
function write_etcd_manifest {

        ### I know this looks fancy, but
        ### Basically, this is just setting up ETCD config file w/ IP Addresses
         cat /vagrant/etcd.manifest | \
        sed "s/NODE_NAME/`hostname`/g" | \
        sed "s/NODE_IP/$IP/g" > /etc/kubernetes/manifests/etcd.manifest
}

### Test of ETCD Members.

function test_etcd {

    echo "----------- DEBUG ------------ KUBELET LOGS -----------------"
    ( journalctl -u kubelet | grep -A 20 -B 20 Fail || echo "no failure in logs")
    echo "----------- END DEBUG OF KUBELET ----------------------------"

    ( curl http://kube0.ha:2379 > /tmp/curl_output || echo "failed etcd!!!" )
    if [ -s /tmp/curl_output ]; then
         echo "etcd success"
    else
         echo "etcd failure.  exit!"
         exit 100
    fi

}

function k8petstore {
     ### run K8petstore .  Should work perfectly IFF flannel and so on is setup properly.
    wget https://raw.githubusercontent.com/kubernetes/kubernetes/release-0.17/examples/k8petstore/k8petstore.sh
     chmod 777 k8petstore.sh
    ./k8petstore.sh
}

function write_api_server_config {
    touch /var/log/kube-apiserver.log
    mkdir -p -m 777 /srv/kubernetes/

    ### We will move files back and forth between the /srv/kube.. directory.
    ### That is how we modulate leader.  Each node will continuously either
    ### ensure that the manifests are in this dir, or else, are in the kubelet manifest dir.
    cp /vagrant/kube-scheduler.manifest /vagrant/kube-controller-manager.manifest /srv/kubernetes

    ### All nodes will run an API Server.  This is because API Server is stateless, so its not a problem
    ### To serve it up everywhere.
    cp /vagrant/kube-apiserver.manifest /etc/kubernetes/manifests/
}

function write_podmaster_config {
    touch /var/log/kube-scheduler.log
    touch /var/log/kube-controller-manager.log

    ### These DO NOT go in manifest. Instead, we mount them here.
    ### We let podmaster swap these in and out of the manifests directory
    ### based on its own internal HA logic.
    cp /vagrant/kube-controller-manager.manifest /srv/kubernetes/
    cp /vagrant/kube-scheduler.manifest /srv/kubernetes/

    #### Finally, the podmaster is the mechanism for election
    cp /vagrant/podmaster.json /etc/kubernetes/manifests/
}

function poll {
    ### wait 10 minutes for kube-apiserver to come online
    for i in `seq 1 600`
    do
           sleep 2
            echo $i
           ### Just testing that the front end comes up.  Not sure how to test total entries etc... (yet)
           ( curl "localhost:8080" > result || echo "failed on attempt $i, retrying again.. api not up yet. " )
           ( cat result || echo "no result" )
           if ( cat result | grep -q api ) ; then
                break
           else
              echo "continue.."
           fi
   done
   if [ $i == 600 ]; then
        exit 2
   fi
}

function install_components {
    ### etcd node - this node only runs etcd in a kubelet, no flannel.
    ### we dont want circular dependency of docker -> flannel -> etcd -> docker
    if [ "`hostname`" == "kube0.ha" ]; then
            write_etcd_manifest
            start_kubelet

            ### precaution to make sure etcd is writable, flush iptables.
             iptables -F
        ### nodes: these will each run their own api server.
    else
        ### Make sure etcd running, flannel needs it.
        test_etcd
        start_kubelet

        ### Flannel setup...
        ### This will restart the kubelet and docker and so on...
        /vagrant/provision-flannel.sh

        echo "Now pulling down flannel nodes. "
        curl -L http://kube0.ha:2379/v2/keys/coreos.com/network/subnets | python -mjson.tool

        echo " Inspect the above lines carefully ^."
        ### All nodes run api server
        write_api_server_config

        ### controller-manager will turn on and off
        ### and same for kube-scheduler
        write_podmaster_config

        # finally, for us to creaet public ips for k8petstore etc, we need the proxy running.
         service kube-proxy start
         service kube-proxy status
    fi
}

initialize
install_components
iptables -F

if [ "`hostname`" == "kube2.ha" ]; then
    poll
    k8petstore
fi

echo "ALL DONE!"
