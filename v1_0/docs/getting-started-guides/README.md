---
layout: docwithnav
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
If you are not sure what OSes and infrastructure is supported, the table below lists all the combinations which have
been tested recently.

For the easiest "kick the tires" experience, please try the [local docker](docker.html) guide.

If you are considering contributing a new guide, please read the
[guidelines](../../docs/devel/writing-a-getting-started-guide.html).

IaaS Provider        | Config. Mgmt | OS     | Networking  | Docs                                              | Conforms | Support Level
-------------------- | ------------ | ------ | ----------  | ---------------------------------------------     | ---------| ----------------------------
GKE                  |              |        | GCE         | [docs](https://cloud.google.com/container-engine) |          | Commercial
Vagrant              | Saltstack    | Fedora | OVS         | [docs](vagrant.html)                                |          | Project
GCE                  | Saltstack    | Debian | GCE         | [docs](gce.html)                                    | [✓][1]   | Project
Azure                | CoreOS       | CoreOS | Weave       | [docs](coreos/azure/README.html)                    |          | Community ([@errordeveloper](https://github.com/errordeveloper), [@squillace](https://github.com/squillace), [@chanezon](https://github.com/chanezon), [@crossorigin](https://github.com/crossorigin))
Docker Single Node   | custom       | N/A    | local       | [docs](docker.html)                                 |          | Project (@brendandburns)
Docker Multi Node    | Flannel      | N/A    | local       | [docs](docker-multinode.html)                       |          | Project (@brendandburns)
Bare-metal           | Ansible      | Fedora | flannel     | [docs](fedora/fedora_ansible_config.html)           |          | Project
Bare-metal           | custom       | Fedora | _none_      | [docs](fedora/fedora_manual_config.html)            |          | Project
Bare-metal           | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.html)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
libvirt              | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.html)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
KVM                  | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.html)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
Mesos/GCE            |              |        |             | [docs](mesos.html)                                  |          | [Community](https://github.com/mesosphere/kubernetes-mesos) ([@jdef](https://github.com/jdef))
AWS                  | CoreOS       | CoreOS | flannel     | [docs](coreos.html)                                 |          | Community
GCE                  | CoreOS       | CoreOS | flannel     | [docs](coreos.html)                                 |          | Community [@pires](https://github.com/pires)
Vagrant              | CoreOS       | CoreOS | flannel     | [docs](coreos.html)                                 |          | Community ( [@pires](https://github.com/pires), [@AntonioMeireles](https://github.com/AntonioMeireles) )
Bare-metal (Offline) | CoreOS       | CoreOS | flannel     | [docs](coreos/bare_metal_offline.html)              |          | Community([@jeffbean](https://github.com/jeffbean))
CloudStack           | Ansible      | CoreOS | flannel     | [docs](cloudstack.html)                             |          | Community (@runseb)
Vmware               |              | Debian | OVS         | [docs](vsphere.html)                                |          | Community (@pietern)
Bare-metal           | custom       | CentOS | _none_      | [docs](centos/centos_manual_config.html)            |          | Community(@coolsvap)
AWS                  | Juju         | Ubuntu | flannel     | [docs](juju.html)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
OpenStack/HPCloud    | Juju         | Ubuntu | flannel     | [docs](juju.html)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
Joyent               | Juju         | Ubuntu | flannel     | [docs](juju.html)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
AWS                  | Saltstack    | Ubuntu | OVS         | [docs](aws.html)                                    |          | Community (@justinsb)
Vmware               | CoreOS       | CoreOS | flannel     | [docs](coreos.html)                                 |          | Community (@kelseyhightower)
Azure                | Saltstack    | Ubuntu | OpenVPN     | [docs](azure.html)                                  |          | Community
Bare-metal           | custom       | Ubuntu | flannel     | [docs](ubuntu.html)                                 |          | Community (@resouer @WIZARD-CXY)
Local                |              |        | _none_      | [docs](locally.html)                                |          | Community (@preillyme)
libvirt/KVM          | CoreOS       | CoreOS | libvirt/KVM | [docs](libvirt-coreos.html)                         |          | Community (@lhuard1A)
oVirt                |              |        |             | [docs](ovirt.html)                                  |          | Community (@simon3z)
Rackspace            | CoreOS       | CoreOS | flannel     | [docs](rackspace.html)                              |          | Community (@doublerr)

Don't see anything above that meets your needs?  See our [Getting Started from Scratch](scratch.html) guide.

*Note*: The above table is ordered by version test/used in notes followed by support level.

Definition of columns:
  - **IaaS Provider** is who/what provides the virtual or physical machines (nodes) that Kubernetes runs on.
  - **OS** is the base operating system of the nodes.
  - **Config. Mgmt** is the configuration management system that helps install and maintain kubernetes software on the
    nodes.
  - **Networking** is what implements the [networking model](../../docs/admin/networking.html).  Those with networking type
    _none_ may not support more than one node, or may support multiple VM nodes only in the same physical node.
  - **Conformance** indicates whether a cluster created with this configuration has passed the project's conformance
    tests for supporting the API and base features of Kubernetes v1.0.0.
  - Support Levels
    - **Project**:  Kubernetes Committers regularly use this configuration, so it usually works with the latest release
      of Kubernetes.
    - **Commercial**: A commercial offering with its own support arrangements.
    - **Community**: Actively supported by community contributions. May not work with more recent releases of kubernetes.
    - **Inactive**: No active maintainer.  Not recommended for first-time K8s users, and may be deleted soon.
  - **Notes** is relevant information such as version k8s used.

<!-- reference style links below here -->
<!-- GCE conformance test result -->
[1]: https://gist.github.com/erictune/4cabc010906afbcc5061


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
