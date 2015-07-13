<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
If you are not sure what OSes and infrastructure is supported, the table below lists all the combinations which have
been tested recently.

For the easiest "kick the tires" experience, please try the [local docker](docker.md) guide.

If you are considering contributing a new guide, please read the
[guidelines](../../docs/devel/writing-a-getting-started-guide.md).

IaaS Provider        | Config. Mgmt | OS     | Networking  | Docs                                                                           | Conforms    | Support Level                | Notes
-------------------- | ------------ | ------ | ----------  | ------------------------------------------------------------------------------ | ----------- | ---------------------------- | -----
GKE                  |              |        | GCE         | [docs](https://cloud.google.com/container-engine)                              | ✓           | Commercial                   | Uses K8s version 0.15.0
Vagrant              | Saltstack    | Fedora | OVS         | [docs](../../docs/getting-started-guides/vagrant.md)                           |             | Project                      | Uses latest via https://get.k8s.io/
GCE                  | Saltstack    | Debian | GCE         | [docs](../../docs/getting-started-guides/gce.md)                               |             | Project                      | Tested with 0.15.0 by @robertbailey
Azure                | CoreOS       | CoreOS | Weave       | [docs](../../docs/getting-started-guides/coreos/azure/README.md)               |             | Community ([@errordeveloper](https://github.com/errordeveloper), [@squillace](https://github.com/squillace), [@chanezon](https://github.com/chanezon), [@crossorigin](https://github.com/crossorigin)) | Uses K8s version 0.17.0
Docker Single Node   | custom       | N/A    | local       | [docs](docker.md)                                                              |             | Project (@brendandburns)     | Tested @ 0.14.1 |
Docker Multi Node    | Flannel      | N/A    | local       | [docs](docker-multinode.md)                                                    |             | Project (@brendandburns)     | Tested @ 0.14.1 |
Bare-metal           | Ansible      | Fedora | flannel     | [docs](../../docs/getting-started-guides/fedora/fedora_ansible_config.md)      |             | Project                      | Uses K8s v0.13.2
Bare-metal           | custom       | Fedora | _none_      | [docs](../../docs/getting-started-guides/fedora/fedora_manual_config.md)       |             | Project                      | Uses K8s v0.13.2
Bare-metal           | custom       | Fedora | flannel     | [docs](../../docs/getting-started-guides/fedora/flannel_multi_node_cluster.md) |             | Community ([@aveshagarwal](https://github.com/aveshagarwal))| Tested with 0.15.0
libvirt              | custom       | Fedora | flannel     | [docs](../../docs/getting-started-guides/fedora/flannel_multi_node_cluster.md) |             | Community ([@aveshagarwal](https://github.com/aveshagarwal))| Tested with 0.15.0
KVM                  | custom       | Fedora | flannel     | [docs](../../docs/getting-started-guides/fedora/flannel_multi_node_cluster.md) |             | Community ([@aveshagarwal](https://github.com/aveshagarwal))| Tested with 0.15.0
Mesos/GCE            |              |        |             | [docs](../../docs/getting-started-guides/mesos.md)                             |             | [Community](https://github.com/mesosphere/kubernetes-mesos) ([@jdef](https://github.com/jdef)) | Uses K8s v0.11.2
AWS                  | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos.md)                            |             | Community                    | Uses K8s version 0.19.3
GCE                  | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos.md)                            |             | Community [@pires](https://github.com/pires) | Uses K8s version 0.19.3
Vagrant              | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos.md)                            |             | Community ( [@pires](https://github.com/pires), [@AntonioMeireles](https://github.com/AntonioMeireles) )           | Uses K8s version 0.19.3
Bare-metal (Offline) | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos/bare_metal_offline.md)         |             | Community([@jeffbean](https://github.com/jeffbean))    | Uses K8s version 0.15.0
CloudStack           | Ansible      | CoreOS | flannel     | [docs](../../docs/getting-started-guides/cloudstack.md)                        |             | Community (@runseb)          | Uses K8s version 0.9.1
Vmware               |              | Debian | OVS         | [docs](../../docs/getting-started-guides/vsphere.md)                           |             | Community (@pietern)         | Uses K8s version 0.9.1
Bare-metal           | custom       | CentOS | _none_      | [docs](../../docs/getting-started-guides/centos/centos_manual_config.md)       |             | Community(@coolsvap)         | Uses K8s v0.9.1
AWS                  | Juju         | Ubuntu | flannel     | [docs](../../docs/getting-started-guides/juju.md)                              |             | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) ) | [Tested](http://reports.vapour.ws/charm-tests-by-charm/kubernetes) K8s v0.8.1
OpenStack/HPCloud    | Juju         | Ubuntu | flannel     | [docs](../../docs/getting-started-guides/juju.md)                              |             | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) ) | [Tested](http://reports.vapour.ws/charm-tests-by-charm/kubernetes) K8s v0.8.1
Joyent               | Juju         | Ubuntu | flannel     | [docs](../../docs/getting-started-guides/juju.md)                              |             | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) ) | [Tested](http://reports.vapour.ws/charm-tests-by-charm/kubernetes) K8s v0.8.1
AWS                  | Saltstack    | Ubuntu | OVS         | [docs](../../docs/getting-started-guides/aws.md)                               |             | Community (@justinsb)        | Uses K8s version 0.5.0
Vmware               | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/coreos.md)                            |             | Community (@kelseyhightower) | Uses K8s version 0.15.0
Azure                | Saltstack    | Ubuntu | OpenVPN     | [docs](../../docs/getting-started-guides/azure.md)                             |             | Community                    |
Bare-metal           | custom       | Ubuntu | flannel     | [docs](../../docs/getting-started-guides/ubuntu.md)                            |             | Community (@resouer @WIZARD-CXY)       | use k8s version 0.19.3
Local                |              |        | _none_      | [docs](../../docs/getting-started-guides/locally.md)                           |             | Community (@preillyme)      |
libvirt/KVM          | CoreOS       | CoreOS | libvirt/KVM | [docs](../../docs/getting-started-guides/libvirt-coreos.md)                    |             | Community (@lhuard1A)       |
oVirt                |              |        |             | [docs](../../docs/getting-started-guides/ovirt.md)                             |             | Community (@simon3z)        |
Rackspace            | CoreOS       | CoreOS | flannel     | [docs](../../docs/getting-started-guides/rackspace.md)                         |             | Community (@doublerr)       | use k8s version 0.18.0

Don't see anything above that meets your needs?  See our [Getting Started from Scratch](../../docs/getting-started-guides/scratch.md) guide.

*Note*: The above table is ordered by version test/used in notes followed by support level.

Definition of columns:
  - **IaaS Provider** is who/what provides the virtual or physical machines (nodes) that Kubernetes runs on.
  - **OS** is the base operating system of the nodes.
  - **Config. Mgmt** is the configuration management system that helps install and maintain kubernetes software on the
    nodes.
  - **Networking** is what implements the [networking model](../../docs/networking.md).  Those with networking type
    _none_ may not support more than one node, or may support multiple VM nodes only in the same physical node.
<<<<<<< Updated upstream
  - **Conformance** indicates whether a cluster created with this configuration has passed the project's conformance
    tests. 
=======
  - **Conforms** indicates whether a cluster created with this configuration has passed the project's conformance
    tests for supporting the API and base features of Kubernetes v1.0.0. 
>>>>>>> Stashed changes
  - Support Levels
    - **Project**:  Kubernetes Committers regularly use this configuration, so it usually works with the latest release
      of Kubernetes.
    - **Commercial**: A commercial offering with its own support arrangements.
    - **Community**: Actively supported by community contributions. May not work with more recent releases of kubernetes.
    - **Inactive**: No active maintainer.  Not recommended for first-time K8s users, and may be deleted soon.
  - **Notes** is relevant information such as version k8s used.

References:
- [1] [GCE conformance test result](https://gist.github.com/erictune/4cabc010906afbcc5061)

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/README.md?pixel)]()
