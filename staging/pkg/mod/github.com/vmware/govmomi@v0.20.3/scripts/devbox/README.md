# devbox

## Overview

This box is an Ubuntu 16.04 VM meant for running development binaries that require a VMX.
For example, the [toolbox][toolbox], [VIC][vic] and [Kubernetes vSphere Cloud Provider][vcp] include code that must run on an ESXi VM.
This script makes it simple to run your laptop/desktop local binaries on such a VM.

This script is a fork of the [VIC devbox](https://github.com/vmware/vic/tree/master/infra/machines/devbox),
without a Vagrant file or provisioning beyond the bento box itself.

[toolbox]:https://github.com/vmware/govmomi/blob/master/toolbox/README.md
[vic]:https://github.com/vmware/vic
[vcp]:https://github.com/kubernetes/kubernetes/tree/master/pkg/cloudprovider/providers/vsphere

## Deployment

Example deployment to ESX:

``` console
% export GOVC_URL=Administrator@vsphere.local:password@vcenter-hostname
% ./create.sh
Deploying to VMware vCenter Server 6.7.0 build-8170161 @ vcenter-hostname...
Converting vagrant box for use with ESXi...
Creating disk 'ubuntu-16.04.vmdk'
  Convert: 100% done.
Virtual disk conversion successful.
Importing vmdk to datastore datastore1...
[08-05-18 13:43:33] Uploading ubuntu-16.04.vmdk... OK
Creating VM dougm-ubuntu-16.04...
Powering on VirtualMachine:7... OK
# For SSH access:
% ssh-add ~/.vagrant.d/insecure_private_key
% ssh vagrant@10.118.66.252
# To NFS export $GOPATH on this host:
% echo "$GOPATH 10.118.66.252(rw,no_subtree_check,sync,all_squash,anonuid=$UID,anongid=$UID)" | sudo tee -a /etc/exports
% sudo service nfs-kernel-server restart
# To NFS mount $GOPATH in the VM:
% ssh vagrant@10.118.66.252 sudo mkdir -p $GOPATH
% ssh vagrant@10.118.66.252 sudo mount 10.118.67.103:$GOPATH $GOPATH
```

## Use Case Examples

Some example use cases for devbox...

### govmomi/toolbox

As an alternative to the CoreOS based [toolbox-test.sh](../../toolbox/toolbox-test.sh), the toolbox can be run on devbox like so:

``` console
% go install github.com/vmware/govmomi/toolbox/toolbox
% ip=$(govc vm.ip -esxcli "$USER-ubuntu-16.04")
% ssh vagrant@$ip sudo service open-vm-tools stop
% ssh vagrant@$ip $GOPATH/bin/toolbox -toolbox.trace
```

### vSphere Integrated Containers

As an alternative to the [VIC devbox](https://github.com/vmware/vic/tree/master/infra/machines/devbox).

### Kubernetes vSphere Cloud Provider

This script builds kubectl and hyperkube locally, generates a Kubernetes cloud provider config for vSphere and runs kubernetes/hack/local-up-cluster.sh from
inside the devbox VM:

``` console
% ./vcp-local-up-cluster.sh
```
