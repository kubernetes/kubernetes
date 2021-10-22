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
# Create a VM and boot stateless ESXi via cdrom/iso

set -o pipefail

usage() {
  cat <<'EOF'
Usage: $0 [-d DISK_GB] [-m MEM_GB] [-i ESX_ISO] [-s] ESX_URL VM_NAME

GOVC_* environment variables also apply, see https://github.com/vmware/govmomi/tree/master/govc#usage
If GOVC_USERNAME is set, it is used to create an account on the ESX vm.  Default is to use the existing root account.
If GOVC_PASSWORD is set, the account password will be set to this value.  Default is to use the given ESX_URL password.
EOF
}

disk=48
mem=16
# 6.7.0U1 https://docs.vmware.com/en/VMware-vSphere/6.7/rn/vsphere-esxi-vcenter-server-67-release-notes.html
iso=VMware-VMvisor-6.7.0-10302608.x86_64.iso

while getopts d:hi:m:s flag
do
  case $flag in
    d)
      disk=$OPTARG
      ;;
    h)
      usage
      exit
      ;;
    i)
      iso=$OPTARG
      ;;
    m)
      mem=$OPTARG
      ;;
    s)
      standalone=true
      ;;
    *)
      usage 1>&2
      exit 1
      ;;
  esac
done

shift $((OPTIND-1))

if [ $# -ne 2 ] ; then
  usage
  exit 1
fi

if [[ "$iso" == *"-Installer-"* ]] ; then
  echo "Invalid iso name (need stateless, not installer): $iso" 1>&2
  exit 1
fi

export GOVC_INSECURE=1
export GOVC_URL=$1
network=${GOVC_NETWORK:-"VM Network"}
username=$GOVC_USERNAME
password=$GOVC_PASSWORD
unset GOVC_USERNAME GOVC_PASSWORD

guest=${GUEST:-"vmkernel65Guest"}

if [ -z "$password" ] ; then
  # extract password from $GOVC_URL
  password=$(govc env GOVC_PASSWORD)
fi

shift

name=$1
shift

echo -n "Checking govc version..."
govc version -require 0.15.0

if [ "$(govc env -x GOVC_URL_HOST)" = "." ] ; then
  if [ "$(uname -s)" = "Darwin" ]; then
    PATH="/Applications/VMware Fusion.app/Contents/Library:$PATH"
  fi

  dir="${name}.vmwarevm"
  vmx="$dir/${name}.vmx"

  if [ -d "$dir" ] ; then
    if vmrun list | grep -q "$vmx" ; then
      vmrun stop "$vmx" hard
    fi
    rm -rf "$dir"
  fi

  mkdir "$dir"
  vmware-vdiskmanager -c -s "${disk}GB" -a lsilogic -t 1 "$dir/${name}.vmdk" 2>/dev/null

  cat > "$vmx" <<EOF
config.version = "8"
virtualHW.version = "11"
numvcpus = "2"
memsize = "$((mem*1024))"
displayName = "$name"
guestOS = "vmkernel6"
vhv.enable = "TRUE"
scsi0.present = "TRUE"
scsi0.virtualDev = "lsilogic"
scsi0:0.present = "TRUE"
scsi0:0.fileName = "${name}.vmdk"
ide1:0.present = "TRUE"
ide1:0.fileName = "$(realpath "$iso")"
ide1:0.deviceType = "cdrom-image"
ethernet0.present = "TRUE"
ethernet0.connectionType = "nat"
ethernet0.virtualDev = "e1000"
ethernet0.wakeOnPcktRcv = "FALSE"
ethernet0.linkStatePropagation.enable = "TRUE"
vmci0.present = "TRUE"
hpet0.present = "TRUE"
tools.syncTime = "TRUE"
pciBridge0.present = "TRUE"
pciBridge4.present = "TRUE"
pciBridge4.virtualDev = "pcieRootPort"
pciBridge4.functions = "8"
pciBridge5.present = "TRUE"
pciBridge5.virtualDev = "pcieRootPort"
pciBridge5.functions = "8"
pciBridge6.present = "TRUE"
pciBridge6.virtualDev = "pcieRootPort"
pciBridge6.functions = "8"
pciBridge7.present = "TRUE"
pciBridge7.virtualDev = "pcieRootPort"
pciBridge7.functions = "8"
EOF

  vmrun start "$vmx" nogui
  vm_ip=$(vmrun getGuestIPAddress "$vmx" -wait)
else
  export GOVC_DATASTORE=${GOVC_DATASTORE:-$(basename "$(govc ls datastore)")}
  if [ "$(govc about -json | jq -r .About.ProductLineId)" == "embeddedEsx" ] ; then
    policy=$(govc host.portgroup.info -json | jq -r ".Portgroup[] | select(.Spec.Name == \"$network\") | .Spec.Policy.Security")
    if [ -n "$policy" ] && [ "$(jq -r <<<"$policy" .AllowPromiscuous)" != "true" ] ; then
      echo "Enabling promiscuous mode for $network on $(govc env -x GOVC_URL_HOST)..."
      govc host.portgroup.change -allow-promiscuous "$network"
    fi
  fi

  boot=$(basename "$iso")
  if ! govc datastore.ls "$boot" > /dev/null 2>&1 ; then
    govc datastore.upload "$iso" "$boot"
  fi

  echo "Creating vm ${name}..."
  govc vm.create -on=false -net "$network" -m $((mem*1024)) -c 2 -g "$guest" -net.adapter=vmxnet3 -disk.controller pvscsi "$name"

  echo "Adding a second nic for ${name}..."
  govc vm.network.add -net "$network" -net.adapter=vmxnet3 -vm "$name"

  echo "Enabling nested hv for ${name}..."
  govc vm.change -vm "$name" -nested-hv-enabled

  echo "Enabling Mac Learning dvFilter for ${name}..."
  seq 0 1 | xargs -I% govc vm.change -vm "$name" \
                  -e ethernet%.filter4.name=dvfilter-maclearn \
                  -e ethernet%.filter4.onFailure=failOpen

  echo "Adding cdrom device to ${name}..."
  id=$(govc device.cdrom.add -vm "$name")

  echo "Inserting $boot into $name cdrom device..."
  govc device.cdrom.insert -vm "$name" -device "$id" "$boot"

  if [ -n "$standalone" ] ; then
    echo "Creating $name disk for use by ESXi..."
    govc vm.disk.create -vm "$name" -name "$name"/disk1 -size "${disk}G"
  fi

  echo "Powering on $name VM..."
  govc vm.power -on "$name"

  echo "Waiting for $name ESXi IP..."
  vm_ip=$(govc vm.ip "$name")

  ! govc events -n 100 "vm/$name" | grep -E 'warning|error'
fi

esx_url="root:@${vm_ip}"
echo "Waiting for $name hostd (via GOVC_URL=$esx_url)..."
while true; do
  if govc about -u "$esx_url" 2>/dev/null; then
    break
  fi

  printf "."
  sleep 1
done

if [ -z "$standalone" ] ; then
  # Create disk for vSAN after boot so they are unclaimed
  echo "Creating $name disks for use by vSAN..."
  govc vm.disk.create -vm "$name" -name "$name"/vsan-cache -size "$((disk/2))G"
  govc vm.disk.create -vm "$name" -name "$name"/vsan-store -size "${disk}G"
fi

# Set target to the ESXi VM
GOVC_URL="$esx_url"

if [ -z "$standalone" ] ; then
  echo "Rescanning ${name} HBA for new devices..."
  disk=($(govc host.storage.info -rescan | grep /vmfs/devices/disks | awk '{print $1}' | sort))

  echo "Marking ${name} disk ${disk[0]} as SSD..."
  govc host.storage.mark -ssd "${disk[0]}"

  echo "Marking ${name} disk ${disk[1]} as HDD..."
  govc host.storage.mark -ssd=false "${disk[1]}"
fi

echo "Configuring NTP for ${name}..."
govc host.date.change -server 0.pool.ntp.org

for id in TSM TSM-SSH ntpd ; do
  printf "Enabling service %s for ${name}...\n" $id
  govc host.service enable $id
  govc host.service start $id
done

if [ -z "$username" ] ; then
  username=root
  action="update"
else
  action="create"
fi

echo "Disabling VSAN device monitoring for ${name}..."
govc host.esxcli system settings advanced set -o /LSOM/VSANDeviceMonitoring -i 0

# A setting of 1 means that vSwp files are created thin, with 0% Object Space Reservation
govc host.esxcli system settings advanced set -o /VSAN/SwapThickProvisionDisabled -i 1
govc host.esxcli system settings advanced set -o /VSAN/FakeSCSIReservations -i 1

echo "ESX host account $action for user $username on ${name}..."
govc host.account.$action -id $username -password "$password"

echo "Granting Admin permissions for user $username on ${name}..."
govc permissions.set -principal $username -role Admin

echo "Enabling guest ARP inspection to get vm IPs without vmtools on ${name}..."
govc host.esxcli system settings advanced set -o /Net/GuestIPHack -i 1

echo "Opening firewall for serial port traffic for ${name}..."
govc host.esxcli network firewall ruleset set -r remoteSerialPort -e true

echo "Setting hostname for ${name}..."
govc host.esxcli system hostname set -H "$name"

echo "Enabling MOB for ${name}..."
govc host.option.set Config.HostAgent.plugins.solo.enableMob true

if which sshpass >/dev/null && [ -e ~/.ssh/id_rsa.pub ] ; then
  echo "Adding ssh authorized key to ${name}..."
  sshpass -p "$password" scp \
          -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=error \
          ~/.ssh/id_rsa.pub "root@$vm_ip:/etc/ssh/keys-root/authorized_keys"
fi

echo "Done: GOVC_URL=${username}:${password}@${vm_ip}"
