#!/bin/bash -e

# Create a Datacenter and connect the underlying ESXi host to vCenter VM

esx_host=$(govc env -x GOVC_URL_HOST)
esx_user=$(govc env GOVC_USERNAME)
esx_pass=$(govc env GOVC_PASSWORD)
vc_ip=${1-$(govc vm.ip "$USER-vcsa")}

unset GOVC_DATACENTER
export GOVC_INSECURE=1 GOVC_URL="Administrator@vsphere.local:${esx_pass}@${vc_ip}"

dc_name=dc1
if [ -z "$(govc ls "/${dc_name}")" ] ; then
  echo "Creating datacenter ${dc_name}..."
  govc datacenter.create "$dc_name"
fi

govc host.add -hostname "$esx_host" -username "$esx_user" -password "$esx_pass" -noverify
