#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

DOCKER_BRIDGE=cbr0
OVS_SWITCH=obr0
DOCKER_OVS_TUN=tun0
TUNNEL_BASE=gre
NETWORK_CONF_PATH=/etc/sysconfig/network-scripts/

# provision network configures the ovs network for pods
function provision-network {
  echo "Verifying network configuration"

  # Only do this operation if the bridge is not defined
  ifconfig | grep -q ${DOCKER_BRIDGE} || {

    echo "It looks like the required network bridge has not yet been created"

    CONTAINER_SUBNETS=(${MASTER_CONTAINER_SUBNET} ${MINION_CONTAINER_SUBNETS[@]})
    CONTAINER_IPS=(${MASTER_IP} ${MINION_IPS[@]})

    # Install openvswitch
    echo "Installing, enabling prerequisites"
    yum install -y openvswitch bridge-utils
    systemctl enable openvswitch
    systemctl start openvswitch

    # create new docker bridge
    echo "Create a new docker bridge"
    ip link set dev ${DOCKER_BRIDGE} down || true
    brctl delbr ${DOCKER_BRIDGE} || true
    brctl addbr ${DOCKER_BRIDGE}
    ip link set dev ${DOCKER_BRIDGE} up
    ifconfig ${DOCKER_BRIDGE} ${CONTAINER_ADDR} netmask ${CONTAINER_NETMASK} up

    # add ovs bridge
    echo "Add ovs bridge"
    ovs-vsctl del-br ${OVS_SWITCH} || true
    ovs-vsctl add-br ${OVS_SWITCH} -- set Bridge ${OVS_SWITCH} fail-mode=secure
    ovs-vsctl set bridge ${OVS_SWITCH} protocols=OpenFlow13
    ovs-vsctl del-port ${OVS_SWITCH} ${TUNNEL_BASE}0 || true
    ovs-vsctl add-port ${OVS_SWITCH} ${TUNNEL_BASE}0 -- set Interface ${TUNNEL_BASE}0 type=${TUNNEL_BASE} options:remote_ip="flow" options:key="flow" ofport_request=10

    # add tun device
    echo "Add tun device"
    ovs-vsctl del-port ${OVS_SWITCH} ${DOCKER_OVS_TUN} || true
    ovs-vsctl add-port ${OVS_SWITCH} ${DOCKER_OVS_TUN} -- set Interface ${DOCKER_OVS_TUN} type=internal ofport_request=9
    brctl addif ${DOCKER_BRIDGE} ${DOCKER_OVS_TUN}
    ip link set ${DOCKER_OVS_TUN} up

    # add oflow rules, because we do not want to use stp
    echo "Add oflow rules"
    ovs-ofctl -O OpenFlow13 del-flows ${OVS_SWITCH}

    # now loop through all other minions and create persistent gre tunnels
    echo "Creating persistent gre tunnels"
    NODE_INDEX=0
    for remote_ip in "${CONTAINER_IPS[@]}"
    do
        if [ "${remote_ip}" == "${NODE_IP}" ]; then
             ovs-ofctl -O OpenFlow13 add-flow ${OVS_SWITCH} "table=0,ip,in_port=10,nw_dst=${CONTAINER_SUBNETS[${NODE_INDEX}]},actions=output:9"
             ovs-ofctl -O OpenFlow13 add-flow ${OVS_SWITCH} "table=0,arp,in_port=10,nw_dst=${CONTAINER_SUBNETS[${NODE_INDEX}]},actions=output:9"
        else
             ovs-ofctl -O OpenFlow13 add-flow ${OVS_SWITCH} "table=0,in_port=9,ip,nw_dst=${CONTAINER_SUBNETS[${NODE_INDEX}]},actions=set_field:${remote_ip}->tun_dst,output:10"
             ovs-ofctl -O OpenFlow13 add-flow ${OVS_SWITCH} "table=0,in_port=9,arp,nw_dst=${CONTAINER_SUBNETS[${NODE_INDEX}]},actions=set_field:${remote_ip}->tun_dst,output:10"
        fi
        ((NODE_INDEX++)) || true
    done
    echo "Created persistent gre tunnels"

    # add ip route rules such that all pod traffic flows through docker bridge and consequently to the gre tunnels
    echo "Add ip route rules such that all pod traffic flows through docker bridge"
    ip route add ${CONTAINER_SUBNET} dev ${DOCKER_BRIDGE} scope link src ${CONTAINER_ADDR}
  }
  echo "Network configuration verified"
}
