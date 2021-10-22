# vCenter cluster testbed automation

## Overview

This directory contains scripts to automate VCSA/ESXi install and cluster configuration for developing and testing.

## Dependencies

### govc

Install the latest release via https://github.com/vmware/govmomi/releases

### jq

Used here to derive static VCSA networking from its parent ESXi host.
But, you should already be using and loving jq for other tasks: http://stedolan.github.io/jq/

## Scripts

### create-esxi-vm.sh

This script creates a VM running stateless ESXi, booted via cdrom/iso.
It will create 2 disks by default:

* vSAN cache disk (Virtual SSD)

* vSAN store disk

The two vSAN disks will be unformatted, leaving them to be autoclaimed
by a vSAN enabled cluster.

Note that for a cluster to use vSAN, it will need at least 3 of these
ESXi VMs.

To create an ESXi VM for standalone use, use the `-s` flag and optionally increase the default disk size with the `-d`
flag:

```
./create-esxi-vm.sh -s -d 56 $GOVC_URL my-esxi-vm
```

The script can also be used directly against Workstation.  Without a username in the url, govc will use local ticket
authentication.  No password is used in this case, but the script will still use this value to set the password for
`root` in the ESX vm.  You may also want to decrease the default disk `-d` and memory `-m` sizes.  Example:

```
GOVC_NETWORK=NAT ./create-esxi-vm.sh -d 16 -m 4 -s :password-for-esx60@localhost $USER-esxbox
```

For use against Fusion, use `.` as the hostname:

```
./create-esxi-vm.sh -d 16 -m 4 -s root:password-for-esx60@. $USER-esxbox
```

### create-vcsa-vm.sh

This script creates a VM with VCSA (Virtual Center Server Appliance) installed.

### create-cluster.sh

The first argument to the script is the IP address of VCSA.
There must be at least three arguments that follow, IP addresses of ESXi hosts, to form the cluster.

The script then creates the following managed objects:

* Datacenter (dc1)

* ClusterComputeResource (cluster1)

* DistributedVirtualSwitch (DSwitch)

* DistributedVirtualPortgroup (PublicNetwork)

* DistributedVirtualPortgroup (InternalNetwork)

All of the given host systems are:

* Added to the ClusterComputeResource (cluster1)

* Added to the DistributedVirtualSwitch (DSwitch)

* Enabled for vSAN traffic (vmk0)

* Firewall configured to enable the remoteSerialPort rule

Cluster configuration includes:

* DRS enabled

* vSAN autoclaim of host system disks (results in shared Datastore "vsanDatastore")

## vSAN Datastore Example

This example will install VCSA, 3 ESXi VMs and create a cluster with vSAN enabled.

```
export GOVC_URL="root:password@some-esx-host"

./create-vcsa-vm.sh -n "${USER}-vcsa" $GOVC_URL

printf "${USER}-esxi-%03d\n" {1..3} | xargs -P3 -n1 ./create-esxi-vm.sh $GOVC_URL

govc vm.ip -k "${USER}-vcsa" "${USER}-esxi-*" | xargs ./create-cluster.sh
```

## Standalone host Example

This example will install VCSA, create a datacenter and connect the underlying ESXi host to the vCenter VM.

```
export GOVC_URL="root:password@some-esx-host"

./create-vcsa-vm.sh -n "${USER}-vcsa" $GOVC_URL

./create-standalone.sh $(govc vm.ip -k "${USER}-vcsa")

```

## Licenses

Optional, if you want to assign licenses to VCSA or ESXi hosts.

Where "ESX_LICENSE" should be that of "vSphere 6 per CPU, Enterprise Plus".

And "VCSA_LICENSE" should be that of "vCenter Server 6, Standard".

```
ESX_LICENSE=... VCSA_LICENSE=... ./assign-licenses.sh
```
