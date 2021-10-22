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
# Create a VCSA VM

usage() {
  echo "Usage: $0 [-n VM_NAME] [-i VCSA_OVA] [-a IP] ESX_URL" 1>&2
  exit 1
}

export GOVC_INSECURE=1

name=vcsa

# 6.7.0U1 https://docs.vmware.com/en/VMware-vSphere/6.7/rn/vsphere-esxi-vcenter-server-67-release-notes.html
ova=VMware-vCenter-Server-Appliance-6.7.0.20000-10244745_OVF10.ova

while getopts a:i:n: flag
do
  case $flag in
    a)
      ip=$OPTARG
      ;;
    i)
      ova=$OPTARG
      ;;
    n)
      name=$OPTARG
      ;;
    *)
      usage
      ;;
  esac
done

if [ -d "$ova" ] ; then
  ova=$(ls "$ova"/*.ovf)
fi

shift $((OPTIND-1))

if [ $# -ne 1 ] ; then
  usage
fi

export GOVC_URL=$1

network=${GOVC_NETWORK:-$(basename "$(govc ls network)")}
product=$(govc about -json | jq -r .About.ProductLineId)
# Use the same password as GOVC_URL
password=$(govc env GOVC_PASSWORD)

if [ -z "$password" ] ; then
  echo "password not set"
  exit 1
fi

opts=(
  cis.vmdir.password=$password
  cis.appliance.root.passwd=$password
  cis.appliance.root.shell=/bin/bash
  cis.deployment.node.type=embedded
  cis.vmdir.domain-name=vsphere.local
  cis.vmdir.site-name=VCSA
  cis.appliance.net.addr.family=ipv4
  cis.appliance.ssh.enabled=True
  cis.ceip_enabled=False
  cis.deployment.autoconfig=True
)

if [ -z "$ip" ] ; then
  mode=dhcp
  ntp=0.pool.ntp.org
else
  mode=static

  # Derive net config from the ESX server
  config=$(govc host.info -k -json | jq -r .HostSystems[].Config)
  gateway=$(jq -r .Network.IpRouteConfig.DefaultGateway <<<"$config")
  dns=$(jq -r .Network.DnsConfig.Address[0] <<<"$config")
  ntp=$(jq -r .DateTimeInfo.NtpConfig.Server[0] <<<"$config")
  route=$(jq -r ".Network.RouteTableInfo.IpRoute[] | select(.DeviceName == \"vmk0\") | select(.Gateway == \"0.0.0.0\")" <<<"$config")
  prefix=$(jq -r .PrefixLength <<<"$route")

  opts+=(cis.appliance.net.addr=$ip
         cis.appliance.net.prefix=$prefix
         cis.appliance.net.dns.servers=$dns
         cis.appliance.net.gateway=$gateway)
fi

opts+=(
  cis.appliance.ntp.servers="$ntp"
  cis.appliance.net.mode=$mode
)

if [ "$product" = "ws" ] ; then
  # workstation does not suport NFC
  dir=$(govc datastore.info -json | jq -r .Datastores[0].Info.Url)

  ovftool --name="$name" --acceptAllEulas "$ova" "$dir"
  vmx="$name/${name}.vmx"
  printf "guestinfo.%s\n" "${opts[@]}" >> "$dir/$vmx"
  govc vm.register "$vmx"
  govc vm.network.change -vm "$name" -net NAT ethernet-0
else
  props=$(printf -- "guestinfo.%s\n" "${opts[@]}" | \
             jq --slurp -R 'split("\n") | map(select(. != "")) | map(split("=")) | map({"Key": .[0], "Value": .[1]})')

  cat <<EOF | govc import.${ova##*.} -options - "$ova"
{
  "Name": "$name",
  "Deployment": "tiny",
  "DiskProvisioning": "thin",
  "IPProtocol": "IPv4",
  "Annotation": "VMware vCenter Server Appliance",
  "PowerOn": false,
  "WaitForIP": false,
  "InjectOvfEnv": true,
  "NetworkMapping": [
    {
      "Name": "Network 1",
      "Network": "${network}"
    }
  ],
  "PropertyMapping": $props
}
EOF
fi

govc vm.change -vm "$name" -g vmwarePhoton64Guest
govc vm.power -on "$name"
govc vm.ip "$name"
