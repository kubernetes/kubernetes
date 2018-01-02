#!/bin/sh

TOP=../../..
CURRENT_DIR=`pwd`
RESOURCES_DIR=$CURRENT_DIR/resources
FUNCTIONAL_DIR=${CURRENT_DIR}/..
HEKETI_DOCKER_IMG=heketi-docker-ci.img
DOCKERDIR=$TOP/extras/docker
CLIENTDIR=$TOP/client/cli/go

source ${FUNCTIONAL_DIR}/lib.sh

### VERSIONS ###
KUBEVERSION=v1.4.3

docker_set_env() {

    ###
    ### CENTOS WORKAROUND ###
    ### Suffering from same bug as https://github.com/getcarina/carina/issues/112
    ###
    if grep -q -s "CentOS\|Fedora" /etc/redhat-release ; then
        echo "CentOS/Fedora DOCKER WORKAROUND"
        curl -sL https://download.getcarina.com/dvm/latest/install.sh | sh
        eval $(minikube docker-env)
        source ~/.dvm/dvm.sh
        dvm install 1.10.3
        dvm use 1.10.3
    else
        eval $(minikube docker-env)
    fi
}

copy_docker_files() {
    docker_set_env
    docker load -i $heketi_docker || fail "Unable to load Heketi docker image"
}

build_docker_file(){
    echo "Create Heketi Docker image"
    heketi_docker=$RESOURCES_DIR/$HEKETI_DOCKER_IMG
    if [ ! -f "$heketi_docker" ] ; then
        cd $DOCKERDIR/ci
        cp $TOP/heketi $DOCKERDIR/ci || fail "Unable to copy $TOP/heketi to $DOCKERDIR/ci"
        _sudo docker build --rm --tag heketi/heketi:ci . || fail "Unable to create docker container"
        _sudo docker save -o $HEKETI_DOCKER_IMG heketi/heketi:ci || fail "Unable to save docker image"
        _sudo chmod 0666 $HEKETI_DOCKER_IMG || fail "Unable to chmod docker image"
        cp $HEKETI_DOCKER_IMG $heketi_docker || fail "Unable to copy image"
        _sudo docker rmi heketi/heketi:ci
        cd $CURRENT_DIR
    fi
    copy_docker_files
}

build_heketi() {
    cd $TOP
    make || fail  "Unable to build heketi"
    cd $CURRENT_DIR
}

copy_client_files() {
    cp $CLIENTDIR/heketi-cli $RESOURCES_DIR || fail "Unable to copy client files"
    cp $TOP/extras/kubernetes/* $RESOURCES_DIR || fail "Unable to copy kubernetes deployment files"
}

teardown() {
    if [ -x /usr/local/bin/minikube ] ; then
        minikube stop > /dev/null
        minikube delete > /dev/null
    fi
    rm -rf $RESOURCES_DIR > /dev/null
}

setup_minikube() {
    if [ ! -d $RESOURCES_DIR ] ; then
        mkdir $RESOURCES_DIR
    fi

    if ! md5sum -c md5sums > /dev/null 2>&1 ; then
        echo -e "\nGet docker-machine"
        curl -Lo docker-machine https://github.com/docker/machine/releases/download/v0.8.2/docker-machine-Linux-x86_64 || fail "Unable to get docker-machine"
        chmod +x docker-machine
        _sudo mv docker-machine /usr/local/bin

        echo -e "\nGet docker-machine-driver-kvm"
        curl -Lo docker-machine-driver-kvm \
            https://github.com/dhiltgen/docker-machine-kvm/releases/download/v0.7.0/docker-machine-driver-kvm || fail "Unable to get docker-machine-driver-kvm"
        chmod +x docker-machine-driver-kvm
        _sudo mv docker-machine-driver-kvm /usr/local/bin

        _sudo usermod -a -G libvirt $(whoami)
        #newgrp libvirt

        echo -e "\nGet minikube"
        curl -Lo minikube \
            https://storage.googleapis.com/minikube/releases/v0.12.2/minikube-linux-amd64 || fail "Unable to get minikube"
        chmod +x minikube
        _sudo mv minikube /usr/local/bin

        echo -e "\nGet kubectl $KUBEVERSION"
        curl -Lo kubectl \
            http://storage.googleapis.com/kubernetes-release/release/${KUBEVERSION}/bin/linux/amd64/kubectl || fail "Unable to get kubectl"
        chmod +x kubectl
        _sudo mv kubectl /usr/local/bin
    fi

    echo -e "\nOpt-out of errors"
    minikube config set WantReportErrorPrompt false
}

start_minikube() {
	minikube start \
		--cpus=2 \
		--memory=2048 \
		--vm-driver=kvm \
		--kubernetes-version="${KUBEVERSION}" || fail "Unable to start minikube"

    # wait until it is ready
    echo -e "\nWait until kubernetes containers are running and ready"
    while [ 3 -ne $(kubectl get pods --all-namespaces | grep Running | wc -l) ] ; do
        echo -n "."
        sleep 1
    done
}

setup() {
    setup_minikube
    build_heketi
    copy_client_files
}

test_teardown() {
    minikube stop
    minikube delete
}

test_setup() {
    start_minikube
    build_docker_file
}


### MAIN ###
teardown
setup

### TESTS ###
for kubetest in test*.sh ; do
   test_setup
   println "TEST $kubetest"
   bash $kubetest; result=$?

   if [ $result -ne 0 ] ; then
       println "FAILED $kubetest"
   else
       println "PASSED $kubetest"
   fi
   test_teardown
done
teardown
exit $result

