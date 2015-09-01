#!/bin/bash

set -ex

# contrail-kubernetes setup and provisioning script. For more info, please refer to
# https://github.com/Juniper/contrail-kubernetes

# Retry a command $RETRY times with $WAIT seconds delay in between
function retry() {
    set +e

    CMD=$1
    echo $CMD
    shift

    if [ -z $RETRY ]; then
        RETRY=10
    fi
    if [ -z $WAIT ]; then
        WAIT=10
    fi

    COUNTER=0
    while [  $COUNTER -lt $RETRY ]; do
        $CMD $*
        if [ "$?" = "0" ]; then
            return
        fi
        let COUNTER=COUNTER+1
        echo Try again $COUNTER/$RETRY
        sleep $WAIT
    done

    set -e

    echo "Error EXIT: $CMD $* Failed with exit code $?"
    exit -1
}

# Run a command in kubernetes-master node
function master() {
    bash -c "sudo $*"
}

# Verify that contrail infra components are up and listening
function verify_contrail_listen_services() {
    RETRY=20
    WAIT=3
    retry master 'netstat -anp | grep LISTEN | grep -w 5672' # RabbitMQ
    retry master 'netstat -anp | grep LISTEN | grep -w 2181' # ZooKeeper
    retry master 'netstat -anp | grep LISTEN | grep -w 9160' # Cassandra
    retry master 'netstat -anp | grep LISTEN | grep -w 5269' # XMPP Server
    retry master 'netstat -anp | grep LISTEN | grep -w 8083' # Control-Node Introspect
    retry master 'netstat -anp | grep LISTEN | grep -w 8443' # IFMAP
    retry master 'netstat -anp | grep LISTEN | grep -w 8082' # API-Server
    retry master 'netstat -anp | grep LISTEN | grep -w 8087' # Schema
    retry master 'netstat -anp | grep LISTEN | grep -w 5998' # discovery
    retry master 'netstat -anp | grep LISTEN | grep -w 8086' # Collector
    retry master 'netstat -anp | grep LISTEN | grep -w 8081' # OpServer
    retry master 'netstat -anp | grep LISTEN | grep -w 8091' # query-engine
    retry master 'netstat -anp | grep LISTEN | grep -w 6379' # redis
    retry master 'netstat -anp | grep LISTEN | grep -w 8143' # WebUI
    retry master 'netstat -anp | grep LISTEN | grep -w 8070' # WebUI
    retry master 'netstat -anp | grep LISTEN | grep -w 3000' # WebUI
}

# Provision controller
function provision_controller() {
    cmd='docker ps | grep contrail-api | grep -v pause | awk "{print \"docker exec \" \$1 \" curl -s https://raw.githubusercontent.com/Juniper/contrail-controller/R2.20/src/config/utils/provision_control.py -o /tmp/provision_control.py\"}" | sudo sh'
    master $cmd
    cmd='docker ps | grep contrail-api | grep -v pause | awk "{print \"docker exec \" \$1 \" curl -s https://raw.githubusercontent.com/Juniper/contrail-controller/R2.20/src/config/utils/provision_bgp.py -o /tmp/provision_bgp.py\"}" | sudo sh'
    master $cmd
    cmd='docker ps | grep contrail-api | grep -v pause | awk "{print \"docker exec \" \$1 \" python /tmp/provision_control.py  --router_asn 64512 --host_name `hostname` --host_ip `hostname --ip-address` --oper add --api_server_ip `hostname --ip-address` --api_server_port 8082\"}" | sudo sh'
    master $cmd
}

# Provision link local service
function provision_linklocal() {
    cmd='docker ps | grep contrail-api | grep -v pause | awk "{print \"docker exec \" \$1 \" curl -s https://raw.githubusercontent.com/Juniper/contrail-controller/R2.20/src/config/utils/provision_linklocal.py -o /tmp/provision_linklocal.py\"}" | sudo sh'
    master $cmd
    cmd='docker ps | grep contrail-api | grep -v pause | awk "{print \"docker exec \" \$1 \" python /tmp/provision_linklocal.py --api_server_ip `hostname --ip-address` --api_server_port 8082 --linklocal_service_name kubernetes --linklocal_service_ip 10.0.0.1 --linklocal_service_port 8080 --ipfabric_service_ip `hostname --ip-` --ipfabric_service_port 8080 --oper add\"}" | sudo sh'
    master $cmd
}

# Setup kube dns endpoints
function setup_kube_dns_endpoints() {
    master kubectl --namespace=kube-system create -f /etc/kubernetes/addons/kube-ui/kube-ui-endpoint.yaml || true
    master kubectl --namespace=kube-system create -f /etc/kubernetes/addons/kube-ui/kube-ui-svc-address.yaml || true
}

# Setup contrail manifest files under kubernetes
function setup_contrail_manifest_files() {
    cmd='wget -qO - https://raw.githubusercontent.com/rombie/contrail-kubernetes/fedora_ubuntu_demo/cluster/manifests.hash | awk "{print \"https://raw.githubusercontent.com/rombie/contrail-kubernetes/fedora_ubuntu_demo/cluster/\"\$1}" | xargs -n1 sudo wget -q --directory-prefix=/etc/contrail/manifests --continue'
    master $cmd

    # Fix kube_network_manager command line args
    SERVICE_CLUSTER_IP_RANGE=${SERVICE_CLUSTER_IP_RANGE:-10.0.0.0/16}
    cmd='sed -i s#--portal_net=10.0.0.0/16#--portal_net='
    cmd="$cmd$SERVICE_CLUSTER_IP_RANGE"
    cmd="$cmd# /etc/contrail/manifests/kube-network-manager.manifest"
    master $cmd

    cmd='grep \"image\": /etc/contrail/manifests/* | cut -d "\"" -f 4 | sort -u | xargs -n1 sudo docker pull'
    RETRY=20
    WAIT=3
    retry master $cmd
    cmd='mv /etc/contrail/manifests/* /etc/kubernetes/manifests/'
    master $cmd

#   cmd='grep source: /srv/salt/contrail-*/* | awk "{print $4}" | xargs -n 1 wget -qO - | grep \"image\": | cut -d "\"" -f 4 | xargs -n1 sudo docker pull'
#   master $cmd
#   master grep source: /srv/salt/contrail-*/* | awk '{print $4}' | xargs -n1 wget -q --directory-prefix=/etc/kubernetes/manifests
}

# Setup contrail-controller components
function setup_contrail_master() {

    # Pull all contrail images and copy the manifest files
    setup_contrail_manifest_files

    # Wait for contrail-control to be ready.
    verify_contrail_listen_services

    # Provision controller
    provision_controller

    # Provision link-local service to connect to kube-api
    provision_linklocal

    # Setip kube-dns
    setup_kube_dns_endpoints
}

setup_contrail_master
