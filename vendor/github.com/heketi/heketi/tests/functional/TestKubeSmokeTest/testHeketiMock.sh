#!/bin/sh

TOP=../../..
CURRENT_DIR=`pwd`
FUNCTIONAL_DIR=${CURRENT_DIR}/..
RESOURCES_DIR=$CURRENT_DIR/resources
PATH=$PATH:$RESOURCES_DIR

source ${FUNCTIONAL_DIR}/lib.sh

# Setup Docker environment
eval $(minikube docker-env)

display_information() {
	# Display information
	echo -e "\nVersions"
	kubectl version

	echo -e "\nDocker containers running"
	docker ps

	echo -e "\nDocker images"
	docker images

	echo -e "\nShow nodes"
	kubectl get nodes
}

setup_heketi() {
	# Start Heketi
	echo -e "\nStart Heketi container"
	kubectl run heketi --image=heketi/heketi:ci --port=8080 || fail "Unable to start heketi container"
	sleep 2

	# This blocks until ready
	kubectl expose deployment heketi --type=NodePort || fail "Unable to expose heketi service"

	echo -e "\nShow Topology"
	export HEKETI_CLI_SERVER=$(minikube service heketi --url)
	heketi-cli topology info

	echo -e "\nLoad mock topology"
	heketi-cli topology load --json=mock-topology.json || fail "Unable to load topology"

	echo -e "\nShow Topology"
	export HEKETI_CLI_SERVER=$(minikube service heketi --url)
	heketi-cli topology info

	echo -e "\nRegister mock endpoints"
	kubectl create -f mock-endpoints.json || fail "Unable to submit mock-endpoints"

	echo -e "\nRegister storage class"
	sed -e \
	"s#%%URL%%#${HEKETI_CLI_SERVER}#" \
	storageclass.yaml.sed > ${RESOURCES_DIR}/sc.yaml
    kubectl create -f ${RESOURCES_DIR}/sc.yaml || fail "Unable to register storage class"
}

test_create() {
	echo "Assert no volumes available"
	if heketi-cli volume list | grep Id ; then
        heketi-cli volume list
		fail "Incorrect number of volumes in Heketi"
	fi

	echo "Submit PVC for 100GiB"
	kubectl create -f pvc.json || fail "Unable to submit PVC"

	sleep 2
	echo "Assert PVC Bound"
	if ! kubectl get pvc | grep claim1 | grep Bound ; then
		fail "PVC is not Bound"
	fi

	echo "Assert only one volume created in Heketi"
	if ! heketi-cli volume list | grep Id | wc -l | grep 1 ; then
		fail "Incorrect number of volumes in Heketi"
	fi

	echo "Assert volume size is 100GiB"
	id=`heketi-cli volume list | grep Id | awk '{print $1}' | cut -d: -f2`
    if ! heketi-cli volume info ${id} | grep Size | cut -d: -f2 | grep 100 ; then
		fail "Invalid size"
	fi
}

test_delete() {
	echo "Delete PVC"
	kubectl delete pvc claim1 || fail "Unable to delete claim1"

    sleep 30
	echo "Assert no volumes available"
	if heketi-cli volume list | grep Id ; then
        heketi-cli volume list
		fail "Incorrect number of volumes in Heketi"
	fi
}

display_information
setup_heketi

echo -e "\n*** Start tests ***"
test_create
test_delete

# Ok now start test




