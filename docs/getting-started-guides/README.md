<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Creating a Kubernetes Cluster

Kubernetes can run on a range of platforms, from your laptop, to VMs on a cloud provider, to rack of
bare metal servers.  The effort required to set up a cluster varies from running a single command to
crafting your own customized cluster.  We'll guide you in picking a solution that fits for your needs.

## Picking the Right Solution

If you just want to "kick the tires" on Kubernetes, we recommend the [local Docker-based](docker.md) solution.

The local Docker-based solution is one of several [Local cluster](#local-machine-solutions) solutions
that are quick to set up, but are limited to running on one machine.

When you are ready to scale up to more machines and higher availability, a [Hosted](#hosted-solutions)
solution is the easiest to create and maintain.

[Turn-key cloud solutions](#turn-key-cloud-solutions) require only a few commands to create
and cover a wider range of cloud providers.

[Custom solutions](#custom-solutions) require more effort to setup but cover and even
they vary from step-by-step instructions to general advice for setting up
a Kubernetes cluster from scratch.

### Local-machine Solutions

Local-machine solutions create a single cluster with one or more Kubernetes nodes on a single
physical machine.  Setup is completely automated and doesn't require a cloud provider account.
But their size and availability is limited to that of a single machine.

The local-machine solutions are:
  - [Local Docker-based](docker.md) (recommended starting point)
  - [Vagrant](vagrant.md) (works on any platform with Vagrant: Linux, MacOS, or Windows.)
  - [No-VM local cluster](locally.md) (Linux only)


### Hosted Solutions

[Google Container Engine](https://cloud.google.com/container-engine) offers managed Kubernetes
clusters.

### Turn-key Cloud Solutions

These solutions allow you to create Kubernetes clusters on a range of Cloud IaaS providers with only a
few commands, and have active community support.
- [GCE](gce.md)
- [AWS](aws.md)
- [Azure](coreos/azure/README.md)

### Custom Solutions

Kubernetes can run on a wide range of Cloud providers and bare-metal environments, and with many
base operating systems.

If you can find a guide below that matches your needs, use it.  It may be a little out of date, but
it will be easier than starting from scratch.  If you do want to start from scratch because you
have special requirements or just because you want to understand what is underneath a Kubernetes
cluster, try the [Getting Started from Scratch](scratch.md) guide.

If you are interested in supporting Kubernetes on a new platform, check out our [advice for
writing a new solution](../../docs/devel/writing-a-getting-started-guide.md).

#### Cloud

These solutions are combinations of cloud provider and OS not covered by the above solutions.
- [AWS + coreos](coreos.md)
- [GCE + CoreOS](coreos.md)
- [AWS + Ubuntu](juju.md)
- [Joyent + Ubuntu](juju.md)
- [Rackspace + CoreOS](rackspace.md)

#### On-Premises VMs

- [Vagrant](coreos.md) (uses CoreOS and flannel)
- [CloudStack](cloudstack.md) (uses Ansible, CoreOS and flannel)
- [Vmware](vsphere.md)  (uses Debian)
- [juju.md](juju.md) (uses Juju, Ubuntu and flannel)
- [Vmware](coreos.md)  (uses CoreOS and flannel)
- [libvirt-coreos.md](libvirt-coreos.md)  (uses CoreOS)
- [oVirt](ovirt.md)
- [libvirt](fedora/flannel_multi_node_cluster.md) (uses Fedora and flannel)
- [KVM](fedora/flannel_multi_node_cluster.md)  (uses Fedora and flannel)

#### Bare Metal

- [Offline](coreos/bare_metal_offline.md) (no internet required.  Uses CoreOS and Flannel)
- [fedora/fedora_ansible_config.md](fedora/fedora_ansible_config.md)
- [Fedora single node](fedora/fedora_manual_config.md)
- [Fedora multi node](fedora/flannel_multi_node_cluster.md)
- [Centos](centos/centos_manual_config.md)
- [Ubuntu](ubuntu.md)
- [Docker Multi Node](docker-multinode.md)

#### Integrations

- [Kubernetes on Mesos](mesos.md) (Uses GCE)

## Table of Solutions

Here are all the solutions mentioned above in table form.

IaaS Provider        | Config. Mgmt | OS     | Networking  | Docs                                              | Conforms | Support Level
-------------------- | ------------ | ------ | ----------  | ---------------------------------------------     | ---------| ----------------------------
GKE                  |              |        | GCE         | [docs](https://cloud.google.com/container-engine) | [✓][3]   | Commercial
Vagrant              | Saltstack    | Fedora | flannel     | [docs](vagrant.md)                                | [✓][2]   | Project
GCE                  | Saltstack    | Debian | GCE         | [docs](gce.md)                                    | [✓][1]   | Project
Azure                | CoreOS       | CoreOS | Weave       | [docs](coreos/azure/README.md)                    |          | Community ([@errordeveloper](https://github.com/errordeveloper), [@squillace](https://github.com/squillace), [@chanezon](https://github.com/chanezon), [@crossorigin](https://github.com/crossorigin))
Docker Single Node   | custom       | N/A    | local       | [docs](docker.md)                                 |          | Project ([@brendandburns](https://github.com/brendandburns))
Docker Multi Node    | Flannel      | N/A    | local       | [docs](docker-multinode.md)                       |          | Project ([@brendandburns](https://github.com/brendandburns))
Bare-metal           | Ansible      | Fedora | flannel     | [docs](fedora/fedora_ansible_config.md)           |          | Project
Digital Ocean        | custom       | Fedora | Calico      | [docs](fedora/fedora-calico.md)                   |          | Community (@djosborne)
Bare-metal           | custom       | Fedora | _none_      | [docs](fedora/fedora_manual_config.md)            |          | Project
Bare-metal           | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.md)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
libvirt              | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.md)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
KVM                  | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.md)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
Mesos/Docker         | custom       | Ubuntu | Docker      | [docs](mesos-docker.md)                           |          | Community ([Kubernetes-Mesos Authors](https://github.com/mesosphere/kubernetes-mesos/blob/master/AUTHORS.md))
Mesos/GCE            |              |        |             | [docs](mesos.md)                                  |          | Community ([Kubernetes-Mesos Authors](https://github.com/mesosphere/kubernetes-mesos/blob/master/AUTHORS.md))
AWS                  | CoreOS       | CoreOS | flannel     | [docs](coreos.md)                                 |          | Community
GCE                  | CoreOS       | CoreOS | flannel     | [docs](coreos.md)                                 |          | Community ([@pires](https://github.com/pires))
Vagrant              | CoreOS       | CoreOS | flannel     | [docs](coreos.md)                                 |          | Community ([@pires](https://github.com/pires), [@AntonioMeireles](https://github.com/AntonioMeireles))
Bare-metal (Offline) | CoreOS       | CoreOS | flannel     | [docs](coreos/bare_metal_offline.md)              |          | Community ([@jeffbean](https://github.com/jeffbean))
Bare-metal           | CoreOS       | CoreOS | Calico      | [docs](coreos/bare_metal_calico.md)               |          | Community ([@caseydavenport](https://github.com/caseydavenport))
CloudStack           | Ansible      | CoreOS | flannel     | [docs](cloudstack.md)                             |          | Community ([@runseb](https://github.com/runseb))
Vmware               |              | Debian | OVS         | [docs](vsphere.md)                                |          | Community ([@pietern](https://github.com/pietern))
Bare-metal           | custom       | CentOS | _none_      | [docs](centos/centos_manual_config.md)            |          | Community ([@coolsvap](https://github.com/coolsvap))
AWS                  | Juju         | Ubuntu | flannel     | [docs](juju.md)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
OpenStack/HPCloud    | Juju         | Ubuntu | flannel     | [docs](juju.md)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
Joyent               | Juju         | Ubuntu | flannel     | [docs](juju.md)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
AWS                  | Saltstack    | Ubuntu | OVS         | [docs](aws.md)                                    |          | Community ([@justinsb](https://github.com/justinsb))
Azure                | Saltstack    | Ubuntu | OpenVPN     | [docs](azure.md)                                  |          | Community
Bare-metal           | custom       | Ubuntu | Calico      | [docs](ubuntu-calico.md)                          |          | Community ([@djosborne](https://github.com/djosborne))
Bare-metal           | custom       | Ubuntu | flannel     | [docs](ubuntu.md)                                 |          | Community ([@resouer](https://github.com/resouer), [@WIZARD-CXY](https://github.com/WIZARD-CXY))
Local                |              |        | _none_      | [docs](locally.md)                                |          | Community ([@preillyme](https://github.com/preillyme))
libvirt/KVM          | CoreOS       | CoreOS | libvirt/KVM | [docs](libvirt-coreos.md)                         |          | Community ([@lhuard1A](https://github.com/lhuard1A))
oVirt                |              |        |             | [docs](ovirt.md)                                  |          | Community ([@simon3z](https://github.com/simon3z))
Rackspace            | CoreOS       | CoreOS | flannel     | [docs](rackspace.md)                              |          | Community ([@doublerr](https://github.com/doublerr))
any                  | any          | any    | any         | [docs](scratch.md)                                |          | Community ([@erictune](https://github.com/erictune))


*Note*: The above table is ordered by version test/used in notes followed by support level.

Definition of columns:
  - **IaaS Provider** is who/what provides the virtual or physical machines (nodes) that Kubernetes runs on.
  - **OS** is the base operating system of the nodes.
  - **Config. Mgmt** is the configuration management system that helps install and maintain Kubernetes software on the
    nodes.
  - **Networking** is what implements the [networking model](../../docs/admin/networking.md).  Those with networking type
    _none_ may not support more than one node, or may support multiple VM nodes only in the same physical node.
  - **Conformance** indicates whether a cluster created with this configuration has passed the project's conformance
    tests for supporting the API and base features of Kubernetes v1.0.0.
  - Support Levels
    - **Project**:  Kubernetes Committers regularly use this configuration, so it usually works with the latest release
      of Kubernetes.
    - **Commercial**: A commercial offering with its own support arrangements.
    - **Community**: Actively supported by community contributions. May not work with more recent releases of Kubernetes.
    - **Inactive**: No active maintainer.  Not recommended for first-time Kubernetes users, and may be deleted soon.
  - **Notes** is relevant information such as the version of Kubernetes used.

<!-- reference style links below here -->
<!-- GCE conformance test result -->
[1]: https://gist.github.com/erictune/4cabc010906afbcc5061
<!-- Vagrant conformance test result -->
[2]: https://gist.github.com/derekwaynecarr/505e56036cdf010bf6b6
<!-- GKE conformance test result -->
[3]: https://gist.github.com/erictune/2f39b22f72565365e59b


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
