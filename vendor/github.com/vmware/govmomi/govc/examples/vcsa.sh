#!/bin/bash -e

lib=$(readlink -nf $(dirname $0))/lib
. $lib/ssh.sh

ova=$1

if [ -z "$ova" ]
then
  ova=./VMware-vCenter-Server-Appliance-5.5.0.10300-2000350_OVF10.ova
fi

# default to local Vagrant esxbox for testing
export GOVC_URL=${GOVC_URL-"https://root:vagrant@localhost:8443/sdk"}

# default VCSA credentials
export GOVC_GUEST_LOGIN=root:vmware

# VM name as defined in the VCSA .ovf
vm_name=VMware_vCenter_Server_Appliance

echo "Importing $ova..."
govc import.ova $ova

echo "Powering on $vm_name..."
govc vm.power -on $vm_name

echo "Waiting for $vm_name's IP address..."
vc=$(govc vm.ip $vm_name)

govc vm.info $vm_name

echo "Uploading ssh public key to $vm_name..."
upload-public-key $vm_name

echo "Configuring vCenter Server Appliance..."

# http://www.virtuallyghetto.com/2012/02/automating-vcenter-server-appliance.html
ssh ${SSH_OPTS} root@$vc <<EOF
echo "Accepting EULA ..."
/usr/sbin/vpxd_servicecfg eula accept

echo "Configuring Embedded DB ..."
/usr/sbin/vpxd_servicecfg db write embedded

echo "Configuring SSO..."
/usr/sbin/vpxd_servicecfg sso write embedded

echo "Starting VCSA ..."
/usr/sbin/vpxd_servicecfg service start
EOF

vc_url=https://${GOVC_GUEST_LOGIN}@${vc}/sdk

echo "VCSA configured and ready..."

govc about -u $vc_url
