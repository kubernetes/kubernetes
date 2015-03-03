#!/bin/bash

set -e
set -o pipefail

BRIDGE_NAME=cbr0  # k8s convention to differentiate from docker0
BRIDGE_MTU=1460  # bytes, a constant for GCE VMs.

# Checks if the bridge name exists, and creates one if necessary.
# Assumes BRIDGE_NAME is set.
function ensure_bridge() {
	if ! ip link show ${BRIDGE_NAME} > /dev/null 2>&1 ; then
		echo "++++ Creating bridge ${BRIDGE_NAME} ..."
		brctl addbr ${BRIDGE_NAME}
	fi
}

# Configures the bridge with IP address and MTU, and brings it up.
# Assumes MINION_IP_RANGE, BRIDGE_NAME and BRIDGE_MTU.
# Assumes the bridge already exists.
function configure_bridge() {
	local parts=(${MINION_IP_RANGE//\// })
	local bridge_ip=${parts[0]/%0/1}
	local length=${parts[1]}

	ip addr add ${bridge_ip}/${length} dev ${BRIDGE_NAME}
	ip link set dev ${BRIDGE_NAME} mtu ${BRIDGE_MTU}
	ip link set dev ${BRIDGE_NAME} up
}

function configure_iptables() {
	iptables -w -t nat -A POSTROUTING -o eth0 -j MASQUERADE ! -d 10.0.0.0/8
}

echo "+++ Ensure the bridge exists ..."
ensure_bridge

echo "+++ Configure the bridge ..."
configure_bridge

echo "+++ Configure iptables for egress ..."
configure_iptables

echo "+++ Done!"

