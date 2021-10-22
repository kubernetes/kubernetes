#!/bin/bash -e

# Deploy Vagrant box to esx

if [ "$(uname -s)" = "Darwin" ]; then
  PATH="/Applications/VMware Fusion.app/Contents/Library:$PATH"
fi

export GOVC_DATASTORE=${GOVC_DATASTORE-"datastore1"}
export GOVC_NETWORK=${GOVC_NETWORK-"VM Network"}
export GOVC_INSECURE=1

echo "Deploying to $(govc object.collect -s - content.about.fullName) @ $(govc env -x GOVC_URL_HOST)..."

box=bento/ubuntu-16.04
provider=$(dirname $box)
name=$(basename $box)
disk="${name}.vmdk"

vm_name=${VM_NAME-"${USER}-${name}"}
vm_memory=${VM_MEMORY-4096}

pushd "$(dirname "$0")" >/dev/null

if ! govc datastore.ls "${name}/${disk}" 1>/dev/null 2>&1 ; then
  if [ ! -e "$disk" ] ; then
    src=$(echo ~/.vagrant.d/boxes/"${provider}"-*-"${name}"/*.*.*/vmware_desktop/disk.vmdk)

    if [ ! -e "$src" ] ; then
      echo "$box not found, install via: vagrant box add --provider vmware_desktop $box"
      exit 1
    fi

    echo "Converting vagrant box for use with ESXi..."
    vmware-vdiskmanager -r "$src" -t 5 "$disk"
  fi

  echo "Importing vmdk to datastore ${GOVC_DATASTORE}..."
  govc import.vmdk "$disk" "$name"
fi

if [ -z "$(govc ls "vm/$vm_name")" ] ; then
  echo "Creating VM ${vm_name}..."

  govc vm.create -m "$vm_memory" -c 2 -g ubuntu64Guest -disk.controller=pvscsi -on=false "$vm_name"

  govc vm.disk.attach -vm "$vm_name" -link=true -disk "$name/$disk"

  govc device.cdrom.add -vm "$vm_name" >/dev/null

  govc vm.change -e disk.enableUUID=1 -vm "$vm_name"

  govc vm.power -on "$vm_name"
  reboot=true
fi

ip=$(govc vm.ip -a -v4 "$vm_name")
me=$(ip route get "$ip" | head -1 | awk '{print $NF}')

ssh-keygen -R "$ip"  1>/dev/null 2>&1
ssh-keyscan -H "$ip" 2>/dev/null >> ~/.ssh/known_hosts

echo "Installing dependencies..."
ssh -i ~/.vagrant.d/insecure_private_key "vagrant@$ip" sudo bash -s - < ./provision.sh

if [ -n "$reboot" ] ; then
  govc vm.power -r "$vm_name"
  ip=$(govc vm.ip -a -v4 "$vm_name")
fi

echo "# For SSH access:"
echo % ssh-add \~/.vagrant.d/insecure_private_key
echo % ssh "vagrant@$ip"

GOPATH=${GOPATH-"$HOME"}
echo "# To NFS export \$GOPATH on this host:"
echo "% echo \"\$GOPATH $ip(rw,no_subtree_check,sync,all_squash,anonuid=\$UID,anongid=\$UID)\" | sudo tee -a /etc/exports"
echo "% sudo service nfs-kernel-server restart"

echo "# To NFS mount \$GOPATH in the VM:"
echo % ssh "vagrant@$ip" sudo mkdir -p "\$GOPATH"
echo % ssh "vagrant@$ip" sudo mount "$me:\$GOPATH" "\$GOPATH"
