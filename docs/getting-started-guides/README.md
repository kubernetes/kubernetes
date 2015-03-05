IaaS Provider  | Config. Mgmt | OS     | Networking  | Docs                                                   | Support Level                | Notes
-------------- | ------------ | ------ | ----------  | ----------------------------------------------------   | ---------------------------- | -----
GCE            | Saltstack    | Debian | GCE         | [docs](../../docs/getting-started-guides/gce.md)       | Project                      | Tested with 0.9.2 by @satnam6502
Vagrant        | Saltstack    | Fedora | OVS         | [docs](../../docs/getting-started-guides/vagrant.md)   | Project                      |
Vagrant        | custom       | Fedora | _none_      | [docs](../../docs/getting-started-guides/fedora/fedora_manual_config.md) | Project    | Uses K8s v0.5-8
Vagrant        | Ansible      | Fedora | flannel     | [docs](../../docs/getting-started-guides/fedora/fedora_ansible_config.md)       | Project    | Uses K8s v0.5-8
GKE            |              |        | GCE         | [docs](https://cloud.google.com/container-engine)      | Commercial                   | Uses K8s version 0.9.2
AWS            | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos.md)    | Community                    | Uses K8s version 0.11.0
GCE            | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos.md)    | Community (@kelseyhightower) | Uses K8s version 0.11.0
Vagrant        | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos.md)    | Community (@pires)           | Uses K8s version 0.11.0
CloudStack     | Ansible      | CoreOS | flannel     | [docs](../../docs/getting-started-guides/cloudstack.md)| Community (@sebgoa)          | Uses K8s version 0.9.1
Vmware         |              | Debian | OVS         | [docs](../../docs/getting-started-guides/vsphere.md)   | Community (@pietern)         | Uses K8s version 0.9.1
AWS            | Saltstack    | Ubuntu | OVS         | [docs](../../docs/getting-started-guides/aws.md)       | Community (@justinsb)        | Uses K8s version 0.5.0
Vmware         | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos.md)    | Community (@kelseyhightower) |
Azure          | Saltstack    | Ubuntu | OpenVPN     | [docs](../../docs/getting-started-guides/azure.md)     | Community (@jeffmendoza)     |
Bare-metal     | custom       | Ubuntu | _none_      | [docs](../../docs/getting-started-guides/ubuntu_single_node.md) | Community (@jainvipin)       |
Bare-metal     | custom       | Ubuntu Cluster | _none_ | [docs](../../docs/getting-started-guides/ubuntu_multinodes_cluster.md) | community (@resouer @WIZARD-CXY) | use k8s version 0.10.1
Local          |              |        | _none_      | [docs](../../docs/getting-started-guides/locally.md)   | Inactive                     |
Ovirt          |              |        |             | [docs](../../docs/getting-started-guides/ovirt.md)     | Inactive                     |
Rackspace      | CoreOS       | CoreOS | Rackspace   | [docs](../../docs/getting-started-guides/rackspace.md) | Inactive                     |
Bare-metal     | custom       | CentOS | _none_      | [docs](../../docs/getting-started-guides/centos/centos_manual_config.md) | Community(@coolsvap)    | Uses K8s v0.9.1
libvirt/KVM    | CoreOS       | CoreOS | libvirt/KVM | [docs](../../docs/getting-started-guides/libvirt-coreos.md) | Community (@lhuard1A)   |
AWS            | Juju         | Ubuntu | flannel     | [docs](../../docs/getting-started-guides/juju.md)      | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) ) | [Tested](http://reports.vapour.ws/charm-tests-by-charm/kubernetes) K8s v0.8.1
OpenStack/HPCloud | Juju      | Ubuntu | flannel     | [docs](../../docs/getting-started-guides/juju.md)      | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) ) | [Tested](http://reports.vapour.ws/charm-tests-by-charm/kubernetes) K8s v0.8.1
Joyent         | Juju         | Ubuntu | flannel     | [docs](../../docs/getting-started-guides/juju.md)      | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) ) | [Tested](http://reports.vapour.ws/charm-tests-by-charm/kubernetes) K8s v0.8.1




Definition of columns:
  - **IaaS Provider** is who/what provides the virtual or physical machines (nodes) that Kubernetes runs on.
  - **OS** is the base operating system of the nodes.
  - **Config. Mgmt** is the configuration management system that helps install and maintain kubernetes software on the
    nodes.
  - **Networking** is what implements the [networking model](../../docs/networking.md).  Those with networking type
    _none_ may not support more than one node, or may support multiple VM nodes only in the same physical node.
  - Support Levels
    - **Project**:  Kubernetes Committers regularly use this configuration, so it usually works with the latest release
      of Kubernetes.
    - **Commercial**: A commercial offering with its own support arrangements.
    - **Community**: Actively supported by community contributions. May not work with more recent releases of kubernetes.
    - **Inactive**: No active maintainer.  Not recommended for first-time K8s users, and may be deleted soon.
