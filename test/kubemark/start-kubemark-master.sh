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
function find-master-pd {
	MASTER_PD_DEVICE=""
	if [[ ! -e /dev/disk/by-id/google-master-pd ]]; then
		return
	fi
	device_info=$(ls -l /dev/disk/by-id/google-master-pd)
	relative_path=${device_info##* }
	MASTER_PD_DEVICE="/dev/disk/by-id/${relative_path}"
}

# Mounts a persistent disk (formatting if needed) to store the persistent data
# on the master -- etcd's data, a few settings, and security certs/keys/tokens.
# safe-format-and-mount only formats an unformatted disk, and mkdir -p will
# leave a directory be if it already exists.
function mount-master-pd {
	find-master-pd
	if [[ -z "${MASTER_PD_DEVICE:-}" ]]; then
		echo "Can't find google-master-pd. Skipping mount."
		return
	fi

	echo "Mounting master-pd"
	local -r pd_path="/dev/disk/by-id/google-master-pd"
	local -r mount_point="/mnt/disks/master-pd"
	# Format and mount the disk, create directories on it for all of the master's
	# persistent data, and link them to where they're used.
	mkdir -p "${mount_point}"
	safe-format-and-mount "${pd_path}" "${mount_point}"
	echo "Mounted master-pd '${pd_path}' at '${mount_point}'"

	# NOTE: These locations on the PD store persistent data, so to maintain
	# upgradeability, these locations should not change.  If they do, take care
	# to maintain a migration path from these locations to whatever new
	# locations.

	# Contains all the data stored in etcd.
	mkdir -m 700 -p "${mount_point}/var/etcd"
	ln -s -f "${mount_point}/var/etcd" /var/etcd
	mkdir -p /etc/srv
	# Contains the dynamically generated apiserver auth certs and keys.
	mkdir -p "${mount_point}/srv/kubernetes"
	ln -s -f "${mount_point}/srv/kubernetes" /etc/srv/kubernetes
	# Directory for kube-apiserver to store SSH key (if necessary).
	mkdir -p "${mount_point}/srv/sshproxy"
	ln -s -f "${mount_point}/srv/sshproxy" /etc/srv/sshproxy

	if ! id etcd &>/dev/null; then
		useradd -s /sbin/nologin -d /var/etcd etcd
	fi
	chown -R etcd "${mount_point}/var/etcd"
	chgrp -R etcd "${mount_point}/var/etcd"
}

function assemble-docker-flags {
	echo "Assemble docker command line flags"
	local docker_opts="-p /var/run/docker.pid --iptables=false --ip-masq=false"
	if [[ "${TEST_CLUSTER:-}" == "true" ]]; then
		docker_opts+=" --log-level=debug"
	else
		docker_opts+=" --log-level=warn"
	fi
	# TODO(shyamjvs): Incorporate network plugin options later.

	# Decide whether to enable a docker registry mirror. This is taken from
	# the "kube-env" metadata value.
	if [[ -n "${DOCKER_REGISTRY_MIRROR_URL:-}" ]]; then
		echo "Enable docker registry mirror at: ${DOCKER_REGISTRY_MIRROR_URL}"
		docker_opts+=" --registry-mirror=${DOCKER_REGISTRY_MIRROR_URL}"
	fi

	echo "DOCKER_OPTS=\"${docker_opts} ${EXTRA_DOCKER_OPTS:-}\"" > /etc/default/docker
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
# Note: This function resembles start-kubelet() function in configure-helper.sh
#       corresponding to kube-up.sh
# TBD [shyamjvs]:
#       1. Do we need to set ulimit for the kubelet?
#       2. Any change in flags to be set for the kubelet?
#          If so, you can set them in cluster/kubemark/config-default.sh
function start-kubelet {
	# Kill any pre-existing kubelet process(es).
	pkill kubelet
	# Replace the builtin kubelet (if any) with the correct binary.
	local builtin_kubelet="$(which kubelet)"
	if [[ -n "${builtin_kubelet}" ]]; then
		cp "${KUBE_BINDIR}/kubelet" "$(dirname "$builtin_kubelet")"
	fi

	# Generate supervisord config for kubelet.
	local name="kubelet"
	local exec_command="${KUBE_BINDIR}/kubelet "
	local flags="${KUBELET_TEST_LOG_LEVEL:-"--v=2"} ${KUBELET_TEST_ARGS:-}"
	flags+=" --allow-privileged=true"
	flags+=" --babysit-daemons=true"
	flags+=" --cgroup-root=/"
	flags+=" --cloud-provider=gce"
	#flags+=" --cluster-dns=${DNS_SERVER_IP}"
	#flags+=" --cluster-domain=${DNS_DOMAIN}"
	flags+=" --config=/etc/kubernetes/manifests"
	#flags+=" --experimental-mounter-path=${KUBE_HOME}/bin/mounter"
	#flags+=" --experimental-check-node-capabilities-before-mount=true"
	if [[ -n "${KUBELET_PORT:-}" ]]; then
		flags+=" --port=${KUBELET_PORT}"
	fi
	# These flags are set because this is a master.
	flags+=" --enable-debugging-handlers=false"
	flags+=" --hairpin-mode=none"
	if [[ ! -z "${KUBELET_APISERVER:-}" && ! -z "${KUBELET_CERT:-}" && ! -z "${KUBELET_KEY:-}" ]]; then
		flags+=" --api-servers=https://${KUBELET_APISERVER}"
		flags+=" --register-schedulable=false"
	else
		flags+=" --pod-cidr=${MASTER_IP_RANGE}"
	fi
	# Network plugin
	if [[ -n "${NETWORK_PROVIDER:-}" ]]; then
		if [[ "${NETWORK_PROVIDER:-}" == "cni" ]]; then
			flags+=" --cni-bin-dir=/home/kubernetes/bin"
		else
			flags+=" --network-plugin-dir=/home/kubernetes/bin"
		fi
		flags+=" --network-plugin=${NETWORK_PROVIDER}"
	fi
	if [[ -n "${NON_MASQUERADE_CIDR:-}" ]]; then
		flags+=" --non-masquerade-cidr=${NON_MASQUERADE_CIDR}"
	fi
	if [[ "${ENABLE_MANIFEST_URL:-}" == "true" ]]; then
		flags+=" --manifest-url=${MANIFEST_URL}"
		flags+=" --manifest-url-header=${MANIFEST_URL_HEADER}"
	fi
	if [[ -n "${ENABLE_CUSTOM_METRICS:-}" ]]; then
		flags+=" --enable-custom-metrics=${ENABLE_CUSTOM_METRICS}"
	fi
	if [[ -n "${NODE_LABELS:-}" ]]; then
		flags+=" --node-labels=${NODE_LABELS}"
	fi
	if [[ -n "${EVICTION_HARD:-}" ]]; then
		flags+=" --eviction-hard=${EVICTION_HARD}"
	fi
	if [[ -n "${FEATURE_GATES:-}" ]]; then
		flags+=" --feature-gates=${FEATURE_GATES}"
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
  cp "${KUBE_ROOT}/etcd.yaml" /etc/kubernetes/manifests

  prepare-log-file /var/log/etcd-events.log
  sed -i -e "s@{{instance_prefix}}@${INSTANCE_PREFIX}@g" "${KUBE_ROOT}/etcd-events.yaml"
  cp "${KUBE_ROOT}/etcd-events.yaml" /etc/kubernetes/manifests
}

# Calculates the following variables based on env variables, which will be used
# by the manifests of several kube-master components.
#   CLOUD_CONFIG_OPT
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
function compute-master-manifest-variables {
	CLOUD_CONFIG_OPT=""
	CLOUD_CONFIG_VOLUME=""
	CLOUD_CONFIG_MOUNT=""
	if [[ -f /etc/gce.conf ]]; then
		CLOUD_CONFIG_OPT="--cloud-config=/etc/gce.conf"
		CLOUD_CONFIG_VOLUME="{\"name\": \"cloudconfigmount\",\"hostPath\": {\"path\": \"/etc/gce.conf\"}},"
		CLOUD_CONFIG_MOUNT="{\"name\": \"cloudconfigmount\",\"mountPath\": \"/etc/gce.conf\", \"readOnly\": true},"
	fi
	DOCKER_REGISTRY="gcr.io/google_containers"
	if [[ -n "${KUBE_DOCKER_REGISTRY:-}" ]]; then
		DOCKER_REGISTRY="${KUBE_DOCKER_REGISTRY}"
	fi
}

# Starts kubernetes apiserver.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute-master-manifest-variables)
#   CLOUD_CONFIG_OPT
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
function start-kube-apiserver {
	echo "Start kubernetes api-server"
	prepare-log-file /var/log/kube-apiserver.log

	# Calculate variables and assemble the command line.
  	local params="${APISERVER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
	params+=" --insecure-bind-address=0.0.0.0"
	params+=" --etcd-servers=http://127.0.0.1:2379"
	params+=" --etcd-servers-overrides=/events#http://127.0.0.1:4002"
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
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
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

########### Main Function ###########
echo "Start to configure master instance for kubemark"

# Extract files from the server tar.
KUBE_ROOT="/home/kubernetes"
cd "${KUBE_ROOT}"
tar xzf kubernetes-server-linux-amd64.tar.gz

# Set required path variables and run config script.
KUBE_HOME="${KUBE_ROOT}/kubernetes"
KUBE_BINDIR="${KUBE_HOME}/server/bin"
sed -i -e 's/\/cluster\/gce//g' "${KUBE_ROOT}/config-default.sh"
source "${KUBE_ROOT}/config-default.sh"

create-dirs
setup-kubelet-dir

# Setup master PD, kubectl (local), docker and load docker images for master componenets.
mount-master-pd
override-kubectl
assemble-docker-flags
load-docker-images

# Start kubelet as a supervisord process and master components as pods.
start-kubelet
compute-master-manifest-variables
start-etcd-servers
start-kube-apiserver
start-kube-controller-manager
start-kube-scheduler

until [ "$(curl 127.0.0.1:8080/healthz 2> /dev/null)" == "ok" ]; do
	sleep 1
done

echo "Done for the configuration for kubermark master"
