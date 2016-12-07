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

# TODO: figure out how to get etcd tag from some real configuration and put it here.

# Define key path variables.
KUBE_ROOT="/home/kubernetes"
KUBE_HOME="${KUBE_ROOT}/kubernetes"
KUBE_BINDIR="${KUBE_HOME}/server/bin"

function create-dirs {
	echo "Creating required directories"
	mkdir -p /var/lib/kubelet
	mkdir -p /etc/kubernetes/manifests
}

# Setup working directory for kubelet.
function setup-kubelet-dir {
	echo "Making /var/lib/kubelet executable for kubelet"
	mount -B /var/lib/kubelet /var/lib/kubelet/
	mount -B -o remount,exec,suid,dev /var/lib/kubelet
}

# Compute etcd/event-etcd related variables.
function compute-etcd-variables {
	EVENT_STORE_IP="${EVENT_STORE_IP:-127.0.0.1}"
	EVENT_STORE_URL="${EVENTS_STORE_URL:-http://${EVENT_STORE_IP}:4002}"
	EVENT_PD="${EVENT_PD:-false}"
	ETCD_IMAGE="${ETCD_IMAGE:-2.2.1}"
	ETCD_QUOTA_BYTES=""
	if [ "${ETCD_VERSION:0:2}" == "3." ]; then
		# TODO: Set larger quota to see if that helps with
		# 'mvcc: database space exceeded' errors. If so, pipe
		# though our setup scripts.
		ETCD_QUOTA_BYTES="--quota-backend-bytes=4294967296 "
	fi
}

# Formats the given device ($1) if needed and mounts it at given mount point
# ($2).
function safe-format-and-mount() {
	device=$1
	mountpoint=$2

	# Format only if the disk is not already formatted.
	if ! tune2fs -l "${device}" ; then
		echo "Formatting '${device}'"
		mkfs.ext4 -F -E lazy_itable_init=0,lazy_journal_init=0,discard "${device}"
	fi

	mkdir -p "${mountpoint}"
	echo "Mounting '${device}' at '${mountpoint}'"
	mount -o discard,defaults "${device}" "${mountpoint}"
}

# Finds the master PD device; returns it in MASTER_PD_DEVICE
function find-master-pd() {
	local -r pd_name=$1
	MASTER_PD_DEVICE=""
	if [[ ! -e /dev/disk/by-id/${pd_name} ]]; then
		return
	fi
	device_info=$(ls -l /dev/disk/by-id/${pd_name})
	relative_path=${device_info##* }
	MASTER_PD_DEVICE="/dev/disk/by-id/${relative_path}"
}

# Mounts a persistent disk (formatting if needed) to store the persistent data
# on the master. safe-format-and-mount only formats an unformatted disk, and
# mkdir -p will leave a directory be if it already exists.
function mount-master-pd() {
	local -r pd_name=$1
	local -r mount_point=$2
	find-master-pd "${pd_name}"

	if [[ -z "${MASTER_PD_DEVICE:-}" ]]; then
		echo "Can't find ${pd_name}. Skipping mount."
		return
	fi

	echo "Mounting master-pd"
	local -r pd_path="/dev/disk/by-id/${pd_name}"
	# Format and mount the disk, create directories on it for all of the master's
	# persistent data, and link them to where they're used.
	mkdir -p "${mount_point}"
	safe-format-and-mount "${pd_path}" "${mount_point}"
	echo "Mounted master-pd '${pd_path}' at '${mount_point}'"

	# NOTE: These locations on the PD store persistent data, so to maintain
	# upgradeability, these locations should not change.  If they do, take care
	# to maintain a migration path from these locations to whatever new
	# locations.
}

function assemble-docker-flags {
	echo "Assemble docker command line flags"
	local docker_opts="-p /var/run/docker.pid --iptables=false --ip-masq=false"
	docker_opts+=" --log-level=debug"  # Since it's a test cluster
	# TODO(shyamjvs): Incorporate network plugin options, etc later.
	echo "DOCKER_OPTS=\"${docker_opts}\"" > /etc/default/docker
	echo "DOCKER_NOFILE=65536" >> /etc/default/docker  # For setting ulimit -n
	service docker restart
	# TODO(shyamjvs): Make docker run through systemd/supervisord.
}

# A helper function for loading a docker image. It keeps trying up to 5 times.
#
# $1: Full path of the docker image
function try-load-docker-image {
	local -r img=$1
	echo "Try to load docker image file ${img}"
	# Temporarily turn off errexit, because we don't want to exit on first failure.
	set +e
	local -r max_attempts=5
	local -i attempt_num=1
	until timeout 30 docker load -i "${img}"; do
		if [[ "${attempt_num}" == "${max_attempts}" ]]; then
			echo "Fail to load docker image file ${img} after ${max_attempts} retries. Exit!!"
			exit 1
		else
			attempt_num=$((attempt_num+1))
			sleep 5
		fi
	done
	# Re-enable errexit.
	set -e
}

# Loads kube-system docker images. It is better to do it before starting kubelet,
# as kubelet will restart docker daemon, which may interfere with loading images.
function load-docker-images {
	echo "Start loading kube-system docker images"
	local -r img_dir="${KUBE_BINDIR}"
	try-load-docker-image "${img_dir}/kube-apiserver.tar"
	try-load-docker-image "${img_dir}/kube-controller-manager.tar"
	try-load-docker-image "${img_dir}/kube-scheduler.tar"
}

# This function assembles the kubelet supervisord config file and starts it using
# supervisorctl, on the kubemark master.
function start-kubelet {
	# Kill any pre-existing kubelet process(es).
	pkill kubelet
	# Replace the builtin kubelet (if any) with the correct binary.
	local -r builtin_kubelet="$(which kubelet)"
	if [[ -n "${builtin_kubelet}" ]]; then
		cp "${KUBE_BINDIR}/kubelet" "$(dirname "$builtin_kubelet")"
	fi

	# Generate supervisord config for kubelet.
	local -r name="kubelet"
	local exec_command="${KUBE_BINDIR}/kubelet "
	local flags="${KUBELET_TEST_LOG_LEVEL:-"--v=2"} ${KUBELET_TEST_ARGS:-}"
	flags+=" --allow-privileged=true"
	flags+=" --babysit-daemons=true"
	flags+=" --cgroup-root=/"
	flags+=" --cloud-provider=gce"
	flags+=" --config=/etc/kubernetes/manifests"
	if [[ -n "${KUBELET_PORT:-}" ]]; then
		flags+=" --port=${KUBELET_PORT}"
	fi
	flags+=" --enable-debugging-handlers=false"
	flags+=" --hairpin-mode=none"
	if [[ ! -z "${KUBELET_APISERVER:-}" && ! -z "${KUBELET_CERT:-}" && ! -z "${KUBELET_KEY:-}" ]]; then
		flags+=" --api-servers=https://${KUBELET_APISERVER}"
		flags+=" --register-schedulable=false"
	else
		flags+=" --pod-cidr=${MASTER_IP_RANGE}"
	fi
	exec_command+="${flags}"

	cat >>"/etc/supervisor/conf.d/${name}.conf" <<EOF
[program:${name}]
command=${exec_command}
stderr_logfile=/var/log/${name}.log
stdout_logfile=/var/log/${name}.log
autorestart=true
startretries=1000000
EOF

	# Update supervisord to make it run kubelet.
	supervisorctl reread
	supervisorctl update
}

# Calculates the following variables based on env variables, which will be used
# by the manifests of several kube-master components.
#   CLOUD_CONFIG_OPT
#   DOCKER_REGISTRY
function compute-master-manifest-variables {
	CLOUD_CONFIG_OPT=""
	DOCKER_REGISTRY="gcr.io/google_containers"
}

# Create the log file and set its properties.
#
# $1 is the file to create.
function prepare-log-file {
	touch $1
	chmod 644 $1
	chown root:root $1
}

# Starts etcd server pod (and etcd-events pod if needed).
# More specifically, it prepares dirs and files, sets the variable value
# in the manifests, and copies them to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute-master-manifest-variables)
#   DOCKER_REGISTRY
function start-etcd-servers {
	echo "Start etcd pods"
	if [[ -d /etc/etcd ]]; then
		rm -rf /etc/etcd
	fi
	if [[ -e /etc/default/etcd ]]; then
		rm -f /etc/default/etcd
	fi
	if [[ -e /etc/systemd/system/etcd.service ]]; then
		rm -f /etc/systemd/system/etcd.service
	fi
	if [[ -e /etc/init.d/etcd ]]; then
		rm -f /etc/init.d/etcd
	fi

	prepare-log-file /var/log/etcd.log
	local src_file="${KUBE_ROOT}/etcd.yaml"
	sed -i -e "s@{{kube_docker_registry}}@${DOCKER_REGISTRY}@g" "${src_file}"
	sed -i -e "s@{{etcd_image}}@${ETCD_IMAGE}@g" "${src_file}"
	sed -i -e "s@{{etcd_quota_bytes}}@${ETCD_QUOTA_BYTES}@g" "${src_file}"
	cp "${src_file}" /etc/kubernetes/manifests

	if [ "${EVENT_STORE_IP:-}" == "127.0.0.1" ]; then
		prepare-log-file /var/log/etcd-events.log
		src_file="${KUBE_ROOT}/etcd-events.yaml"
		sed -i -e "s@{{instance_prefix}}@${INSTANCE_PREFIX}@g" "${src_file}"
		sed -i -e "s@{{kube_docker_registry}}@${DOCKER_REGISTRY}@g" "${src_file}"
		sed -i -e "s@{{etcd_image}}@${ETCD_IMAGE}@g" "${src_file}"
		sed -i -e "s@{{etcd_quota_bytes}}@${ETCD_QUOTA_BYTES}@g" "${src_file}"
		cp "${src_file}" /etc/kubernetes/manifests
	fi
}

# Starts kubernetes apiserver.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute-master-manifest-variables)
#   CLOUD_CONFIG_OPT
#   DOCKER_REGISTRY
function start-kube-apiserver {
	echo "Start kubernetes api-server"
	prepare-log-file /var/log/kube-apiserver.log

	# Calculate variables and assemble the command line.
	local params="${APISERVER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
	params+=" --insecure-bind-address=0.0.0.0"
	params+=" --etcd-servers=http://127.0.0.1:2379"
	params+=" --etcd-servers-overrides=/events#${EVENT_STORE_URL}"
	params+=" --tls-cert-file=/srv/kubernetes/server.cert"
	params+=" --tls-private-key-file=/srv/kubernetes/server.key"
	params+=" --client-ca-file=/srv/kubernetes/ca.crt"
	params+=" --token-auth-file=/srv/kubernetes/known_tokens.csv"
	params+=" --secure-port=443"
	params+=" --basic-auth-file=/srv/kubernetes/basic_auth.csv"
	params+=" --target-ram-mb=$((${NUM_NODES} * 60))"
	params+=" --storage-backend=${STORAGE_BACKEND}"
	params+=" --service-cluster-ip-range=${SERVICE_CLUSTER_IP_RANGE}"
	if [ -z "${CUSTOM_ADMISSION_PLUGINS:-}" ]; then
		params+=" --admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,ResourceQuota"
	else
		params+=" --admission-control=${CUSTOM_ADMISSION_PLUGINS}"
	fi
	local -r kube_apiserver_docker_tag=$(cat ${KUBE_BINDIR}/kube-apiserver.docker_tag)
	local -r src_file="${KUBE_ROOT}/kube-apiserver.yaml"

	# Evaluate variables.
	sed -i -e "s@{{params}}@${params}@g" "${src_file}"
	sed -i -e "s@{{kube_docker_registry}}@${DOCKER_REGISTRY}@g" "${src_file}"
	sed -i -e "s@{{kube-apiserver_docker_tag}}@${kube_apiserver_docker_tag}@g" "${src_file}"
	cp "${src_file}" /etc/kubernetes/manifests
}

# Starts kubernetes controller manager.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute-master-manifest-variables)
#   CLOUD_CONFIG_OPT
#   DOCKER_REGISTRY
function start-kube-controller-manager {
	echo "Start kubernetes controller-manager"
	prepare-log-file /var/log/kube-controller-manager.log

	# Calculate variables and assemble the command line.
	local params="${CONTROLLER_MANAGER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
	params+=" --master=127.0.0.1:8080"
	params+=" --service-account-private-key-file=/srv/kubernetes/server.key"
	params+=" --root-ca-file=/srv/kubernetes/ca.crt"
	params+=" --allocate-node-cidrs=${ALLOCATE_NODE_CIDRS}"
	params+=" --cluster-cidr=${CLUSTER_IP_RANGE}"
	params+=" --service-cluster-ip-range=${SERVICE_CLUSTER_IP_RANGE}"
	params+=" --terminated-pod-gc-threshold=${TERMINATED_POD_GC_THRESHOLD}"
	local -r kube_rc_docker_tag=$(cat ${KUBE_BINDIR}/kube-controller-manager.docker_tag)
	local -r src_file="${KUBE_ROOT}/kube-controller-manager.yaml"

	# Evaluate variables.
	sed -i -e "s@{{params}}@${params}@g" "${src_file}"
	sed -i -e "s@{{kube_docker_registry}}@${DOCKER_REGISTRY}@g" "${src_file}"
	sed -i -e "s@{{kube-controller-manager_docker_tag}}@${kube_rc_docker_tag}@g" "${src_file}"
	cp "${src_file}" /etc/kubernetes/manifests
}

# Starts kubernetes scheduler.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in compute-master-manifest-variables)
#   DOCKER_REGISTRY
function start-kube-scheduler {
	echo "Start kubernetes scheduler"
	prepare-log-file /var/log/kube-scheduler.log

	# Calculate variables and set them in the manifest.
	params="${SCHEDULER_TEST_ARGS:-}"
	params+=" --master=127.0.0.1:8080"
	local -r kube_scheduler_docker_tag=$(cat "${KUBE_BINDIR}/kube-scheduler.docker_tag")
	local -r src_file="${KUBE_ROOT}/kube-scheduler.yaml"

	# Evaluate variables.
	sed -i -e "s@{{params}}@${params}@g" "${src_file}"
	sed -i -e "s@{{kube_docker_registry}}@${DOCKER_REGISTRY}@g" "${src_file}"
	sed -i -e "s@{{kube-scheduler_docker_tag}}@${kube_scheduler_docker_tag}@g" "${src_file}"
	sed -i -e "s@{{instance_prefix}}@${INSTANCE_PREFIX}@g" "${src_file}"
	cp "${src_file}" /etc/kubernetes/manifests
}

function override-kubectl {
	echo "overriding kubectl"
	echo "export PATH=${KUBE_BINDIR}:\$PATH" > /etc/profile.d/kube_env.sh
}

############################### Main Function ########################################
echo "Start to configure master instance for kubemark"

# Extract files from the server tar.
cd "${KUBE_ROOT}"
tar xzf kubernetes-server-linux-amd64.tar.gz

# Run config script for environment variables.
sed -i -e 's/\/cluster\/gce//g' "${KUBE_ROOT}/config-default.sh"
source "${KUBE_ROOT}/config-default.sh"

# Setup directory structure and environment variables for master components.
create-dirs
setup-kubelet-dir
compute-etcd-variables
compute-master-manifest-variables
override-kubectl

# Mount master PD for etcd and create symbolic links to it.
{
	main_etcd_mount_point="/mnt/disks/master-pd"
	mount-master-pd "google-master-pd" "${main_etcd_mount_point}"
	# Contains all the data stored in etcd.
	mkdir -m 700 -p "${main_etcd_mount_point}/var/etcd"
	ln -s -f "${main_etcd_mount_point}/var/etcd" /var/etcd
	mkdir -p /etc/srv
	# Contains the dynamically generated apiserver auth certs and keys.
	mkdir -p "${main_etcd_mount_point}/srv/kubernetes"
	ln -s -f "${main_etcd_mount_point}/srv/kubernetes" /etc/srv/kubernetes
	# Directory for kube-apiserver to store SSH key (if necessary).
	mkdir -p "${main_etcd_mount_point}/srv/sshproxy"
	ln -s -f "${main_etcd_mount_point}/srv/sshproxy" /etc/srv/sshproxy
}

# Mount master PD for event-etcd (if required) and create symbolic links to it.
{
	if [ "${EVENT_PD:-false}" == "true" ]; then
		event_etcd_mount_point="/mnt/disks/master-event-pd"
		mount-master-pd "google-master-event-pd" "${event_etcd_mount_point}"
		# Contains all the data stored in event etcd.
		mkdir -m 700 -p "${event_etcd_mount_point}/var/etcd/events"
		ln -s -f "${event_etcd_mount_point}/var/etcd/events" /var/etcd/events
	fi
}

# Setup docker flags and load images of the master components.
assemble-docker-flags
load-docker-images

# Start kubelet as a supervisord process and master components as pods.
start-kubelet
start-etcd-servers
start-kube-apiserver
start-kube-controller-manager
start-kube-scheduler

# Wait till all components are working fine.
until [ "$(curl 127.0.0.1:8080/healthz 2> /dev/null)" == "ok" ]; do
	sleep 1
done

echo "Done for the configuration for kubermark master"
