## What is oVirt

oVirt is a virtual datacenter manager that delivers powerful management of multiple virtual machines on multiple hosts. Using KVM and libvirt, oVirt can be installed on Fedora, CentOS, or Red Hat Enterprise Linux hosts to set up and manage your virtual data center.

## oVirt Cloud Provider Deployment

The oVirt cloud provider allows to easily discover and automatically add new VM instances as nodes to your kubernetes cluster.
At the moment there are no community-supported or pre-loaded VM images including kubernetes but it is possible to [import] or [install] Project Atomic (or Fedora) in a VM to [generate a template]. Any other distribution that includes kubernetes may work as well.

It is mandatory to [install the ovirt-guest-agent] in the guests for the VM ip address and hostname to be reported to ovirt-engine and ultimately to kubernetes.

Once the kubernetes template is available it is possible to start instantiating VMs that can be discovered by the cloud provider.

[import]: http://ovedou.blogspot.it/2014/03/importing-glance-images-as-ovirt.html
[install]: http://www.ovirt.org/Quick_Start_Guide#Create_Virtual_Machines
[generate a template]: http://www.ovirt.org/Quick_Start_Guide#Using_Templates
[install the ovirt-guest-agent]: http://www.ovirt.org/How_to_install_the_guest_agent_in_Fedora

## Using the oVirt Cloud Provider

The oVirt Cloud Provider requires access to the oVirt REST-API to gather the proper information, the required credential should be specified in the `ovirt-cloud.conf` file:

    [connection]
    uri = https://localhost:8443/ovirt-engine/api
    username = admin@internal
    password = admin

In the same file it is possible to specify (using the `filters` section) what search query to use to identify the VMs to be reported to kubernetes:

    [filters]
    # Search query used to find nodes
    vms = tag=kubernetes

In the above example all the VMs tagged with the `kubernetes` label will be reported as nodes to kubernetes.

The `ovirt-cloud.conf` file then must be specified in kube-controller-manager:

    kube-controller-manager ... --cloud-provider=ovirt --cloud-config=/path/to/ovirt-cloud.conf ...

## oVirt Cloud Provider Screencast

This short screencast demonstrates how the oVirt Cloud Provider can be used to dynamically add VMs to your kubernetes cluster.

[![Screencast](http://img.youtube.com/vi/JyyST4ZKne8/0.jpg)](http://www.youtube.com/watch?v=JyyST4ZKne8)
