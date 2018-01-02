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

create_fake_application() {
	pod=$1
	app=$2
	kubectl exec $pod -- sh -c "echo '#!/bin/sh' > /bin/${app}" || fail "Unable to create /bin/${app}"
	kubectl exec $pod -- chmod +x /bin/${app} || fail "Unable to chmod +x /bin/${app}"
}

create_bash() {
	pod=$1
	app="bash"
	kubectl exec $pod -- sh -c "echo '#!/bin/sh' > /bin/${app}" || fail "Unable to create /bin/${app}"
	kubectl exec $pod -- sh -c "echo 'sh \$@' >> /bin/${app}" || fail "Unable to add to /bin/${app}"
	kubectl exec $pod -- chmod +x /bin/${app} || fail "Unable to chmod +x /bin/${app}"
}

create_fake_vgdisplay() {
	pod=$1
	app=vgdisplay
	kubectl exec $pod -- sh -c "echo '#!/bin/sh' > /bin/${app}" || fail "Unable to create /bin/${app}"

	## This pretends that there is a disk with 99G of free space and 0 used.
	kubectl exec $pod -- sh -c "echo 'echo mock:r/w:772:-1:0:0:0:-1:0:1:1:104722432:4096:25567:0:25567:3xn8HX-x6cB-CJJy-Sj6Q-I0CY-gJLY-qlipv5' >> /bin/${app}" || fail "Unable to add to /bin/${app}"
	kubectl exec $pod -- chmod +x /bin/${app} || fail "Unable to chmod +x /bin/${app}"
}

start_mock_gluster_container() {
	# Use a busybox container
	  kubectl run gluster$1 \
		--restart=Never \
		--image=busybox \
		--labels=glusterfs-node=daemonset \
		--command -- sleep 10000 || fail "Unable to start gluster$1"

	# Wait until it is running
	while ! kubectl get pods | grep gluster$1 | grep "1/1" > /dev/null ; do
		sleep 1
	done

	# Create fake gluster file
	create_fake_application gluster$1 "gluster"
	create_fake_application gluster$1 "pvcreate"
	create_fake_application gluster$1 "vgcreate"
	create_fake_application gluster$1 "pvremove"
	create_fake_application gluster$1 "vgremove"
	create_fake_vgdisplay gluster$1
	create_bash gluster$1
}

setup_all_pods() {

	kubectl get nodes --show-labels

	# Start Heketi
	echo -e "\nStart Heketi container"
    sed -e "s#heketi/heketi:dev#heketi/heketi:ci#" \
        -e "s#Always#IfNotPresent#" \
        $RESOURCES_DIR/deploy-heketi-deployment.json | kubectl create -f -

	# Wait until it is running
	echo "Wait until deploy-heketi is ready"
	while ! kubectl get pods | grep heketi | grep "1/1" > /dev/null ; do
		echo -n "."
		sleep 1
	done

	echo "Delete the cluster service because it cannot be used in minikube"
	kubectl delete service deploy-heketi

	echo "Create a service for deploy-heketi"
	kubectl expose deployment deploy-heketi --port=8080 --type=NodePort || fail "Unable to expose heketi service"

	echo -e "\nShow Topology"
	export HEKETI_CLI_SERVER=$(minikube service deploy-heketi --url)
	heketi-cli topology info

	echo -e "\nStart gluster mock container"
	start_mock_gluster_container 1
}

test_add_devices() {
	echo -e "\nGet the Heketi server connection"
	heketi-cli cluster create || fail "Unable to create cluster"

	CLUSTERID=$(heketi-cli cluster list | sed -e '$!d')

	echo -e "\nAdd Node"
	heketi-cli node add --zone=1 --cluster=$CLUSTERID \
		--management-host-name=minikube --storage-host-name=minikube || fail "Unable to add gluster1"

	echo -e "\nAdd device"
	nodeid=$(heketi-cli node list | awk '{print $1}' | awk -F: '{print $2}')
	heketi-cli device add --name=/dev/fakedevice --node=$nodeid || fail "Unable to add device"

	# Check sizes
	device_size=$(heketi-cli topology info | grep fakedevice | awk '{print $5}' | cut -d: -f2)
	device_used=$(heketi-cli topology info | grep fakedevice | awk '{print $7}' | cut -d: -f2)
	device_free=$(heketi-cli topology info | grep fakedevice | awk '{print $9}' | cut -d: -f2)
	if [ 99 -ne $device_size ] ; then
		fail "Expected size of 99 instead got $device_size"
	fi
	if [ 0 -ne $device_used ] ; then
		fail "Expected used of 0 instead got $device_used"
	fi
	if [ 99 -ne $device_free ] ; then
		fail "Expected free of 99 instead got $device_free"
	fi

	echo -e "\nShow Topology"
	heketi-cli topology info
}

display_information
setup_all_pods

echo -e "\n*** Start tests ***"
test_add_devices

# Ok now start test
