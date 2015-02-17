IaaS Provider  | Config. Mgmt | OS     | Docs                                                   | Support Level                | Notes
-------------- | ------------ | ------ | ----------------------------------------------------   | ---------------------------- | -----
GCE            | Saltstack    | Debian | [docs](../../docs/getting-started-guides/gce.md)       | Project                      | Tested with 0.9.2 by @satnam6502
Vagrant        | Saltstack    | Fedora | [docs](../../docs/getting-started-guides/vagrant.md)   | Project                      |
Vagrant        | custom       | Fedora | [docs](../../docs/getting-started-guides/fedora/fedora_manual_config.md) | Project    | Uses K8s v0.5-8
Vagrant        | Ansible      | Fedora | [docs](../../docs/getting-started-guides/fedora/fedora_ansible.md)       | Project    | Uses K8s v0.5-8
GKE            |              |        | [docs](https://cloud.google.com/container-engine)      | Commercial                   | Uses K8s version 0.9.2
AWS            | CoreOS       | CoreOS | [docs](../../docs/getting-started-guides/coreos.md)    | Community                    | Uses K8s version 0.10.1
GCE            | CoreOS       | CoreOS | [docs](../../docs/getting-started-guides/coreos.md)    | Community (@kelseyhightower) | Uses K8s version 0.10.1
Vagrant        | CoreOS       | CoreOS | [docs](../../docs/getting-started-guides/coreos.md)    | Community (@pires)           | Uses K8s version 0.10.1
CloudStack     | Ansible      | CoreOS | [docs](../../docs/getting-started-guides/cloudstack.md)| Community (@sebgoa)          | Uses K8s version 0.9.1
Vmware         |              | Debian | [docs](../../docs/getting-started-guides/vsphere.md)   | Community (@pietern)         | Uses K8s version 0.9.1
AWS            | Saltstack    | Ubuntu | [docs](../../docs/getting-started-guides/aws.md)       | Community (@justinsb)        | Uses K8s version 0.5.0
Vmware         | CoreOS       | CoreOS | [docs](../../docs/getting-started-guides/coreos.md)    | Community (@kelseyhightower) |
Azure          | Saltstack    | Ubuntu | [docs](../../docs/getting-started-guides/azure.md)     | Community (@jeffmendoza)     |
Bare-metal     | custom       | Ubuntu | [docs](../../docs/getting-started-guides/ubuntu_single_node.md) | Community (@jainvipin)       |
Local          |              |        | [docs](../../docs/getting-started-guides/locally.md)   | Inactive                     |
Ovirt          |              |        | [docs](../../docs/getting-started-guides/ovirt.md)     | Inactive                     |
Rackspace      | CoreOS       | CoreOS | [docs](../../docs/getting-started-guides/rackspace.md) | Inactive                     |
Bare-metal     | custom       | CentOS | [docs](../../docs/getting-started-guides/centos/centos_manual_config.md) | Community(@coolsvap)    | Uses K8s v0.9.1
libvirt/KVM    | CoreOS       | CoreOS | [docs](../../docs/getting-started-guides/libvirt-coreos.md) | Community (@lhuard1A)   |
Definition of columns:
  - **IaaS Provider** is who/what provides the virtual or physical machines (nodes) that Kubernetes runs on.
  - **OS** is the base operating system of the nodes.
  - **Config. Mgmt** is the configuration management system that helps install and maintain kubernetes software on the
    nodes.
  - Support Levels
    - **Project**:  Kubernetes Committers regularly use this configuration, so it usually works with the latest release
      of Kubernetes.
    - **Commercial**: A commercial offering with its own support arrangements.
    - **Community**: Actively supported by community contributions. May not work with more recent releases of kubernetes.
    - **Inactive**: No active maintainer.  Not recommended for first-time K8s users, and may be deleted soon.
