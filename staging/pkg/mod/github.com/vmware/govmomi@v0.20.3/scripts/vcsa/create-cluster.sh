#!/bin/bash -e

# Copyright 2017-2018 VMware, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Configure a vCenter cluster with vSAN datastore, DVS and DVPGs

export GOVC_INSECURE=1
export GOVC_USERNAME=${GOVC_USERNAME:-"Administrator@vsphere.local"}
if [ -z "$GOVC_PASSWORD" ] ; then
  # extract password from $GOVC_URL
  GOVC_PASSWORD=$(govc env GOVC_PASSWORD)
fi

usage() {
  echo "Usage: $0 [-d DATACENTER] [-c CLUSTER] VCSA_IP ESX_IP..." 1>&2
  exit 1
}

# Defaults
dc_name="dc1"
cluster_name="cluster1"
vsan_vnic="vmk0"

while getopts c:d: flag
do
  case $flag in
    c)
      cluster_name=$OPTARG
      ;;
    d)
      dc_name=$OPTARG
      ;;
    *)
      usage
      ;;
  esac
done

shift $((OPTIND-1))

if [ $# -lt 2 ] ; then
  usage
fi

vc_ip=$1
shift

unset GOVC_DATACENTER
export GOVC_URL="${GOVC_USERNAME}:${GOVC_PASSWORD}@${vc_ip}"

cluster_path="/$dc_name/host/$cluster_name"
dvs_path="/$dc_name/network/DSwitch"
public_network="/$dc_name/network/PublicNetwork"
internal_network="/$dc_name/network/InternalNetwork"

if [ -z "$(govc ls "/$dc_name")" ] ; then
  echo "Creating datacenter ${dc_name}..."
  govc datacenter.create "$dc_name"
fi

export GOVC_DATACENTER="$dc_name"

if [ -z "$(govc ls "$cluster_path")" ] ; then
  echo "Creating cluster ${cluster_path}..."
  govc cluster.create "$cluster_name"
fi

if [ -z "$(govc ls "$dvs_path")" ] ; then
  echo "Creating dvs ${dvs_path}..."
  govc dvs.create -product-version 6.0.0 -folder "$(dirname "$dvs_path")" "$(basename "$dvs_path")"
fi

if [ -z "$(govc ls "$public_network")" ] ; then
  govc dvs.portgroup.add -dvs "$dvs_path" -type earlyBinding -nports 16 "$(basename "$public_network")"
fi

if [ -z "$(govc ls "$internal_network")" ] ; then
  govc dvs.portgroup.add -dvs "$dvs_path" -type ephemeral "$(basename "$internal_network")"
fi

hosts=()
vsan_hosts=()

for host_ip in "$@" ; do
  host_path="$cluster_path/$host_ip"
  hosts+=($host_path)

  if [ -z "$(govc ls "$host_path")" ] ; then
    echo "Adding host ($host_ip) to cluster $cluster_name"
    govc cluster.add -cluster "$cluster_path" -noverify -force \
         -hostname "$host_ip" -username root -password "$GOVC_PASSWORD"
  fi

  unclaimed=$(govc host.storage.info -host "$host_path" -unclaimed | tail -n+2 | wc -l)
  if [ "$unclaimed" -eq 2 ] ; then
    echo "Enabling vSAN traffic on ${vsan_vnic} for ${host_path}..."
    govc host.vnic.service -host "$host_path" -enable vsan "$vsan_vnic"
    vsan_hosts+=($host_path)
  else
    echo "Skipping vSAN configuration for ${host_path}: $unclaimed unclaimed disks"
  fi
done

govc dvs.add -dvs "$dvs_path" -pnic vmnic1 "${hosts[@]}"

echo "Enabling DRS for ${cluster_path}..."
govc cluster.change -drs-enabled "$cluster_path"

if [ ${#vsan_hosts[@]} -ge 3 ] ; then
  echo "Enabling vSAN for ${cluster_path}..."
  govc cluster.change -vsan-enabled -vsan-autoclaim "$cluster_path"
fi

echo "Enabling HA for ${cluster_path}..."
govc cluster.change -ha-enabled "$cluster_path"

echo "Granting Admin permissions for user root..."
govc permissions.set -principal root -role Admin

echo "Done."
