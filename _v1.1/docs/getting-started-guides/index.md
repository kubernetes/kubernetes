---
layout: docwithnav
title: "Creating a Kubernetes Cluster"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

Creating a Kubernetes Cluster
----------------------------------------

Kubernetes can run on a range of platforms, from your laptop, to VMs on a cloud provider, to rack of
bare metal servers.  The effort required to set up a cluster varies from running a single command to
crafting your own customized cluster.  We'll guide you in picking a solution that fits for your needs.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

  - [Picking the Right Solution](#picking-the-right-solution)
    - [Local-machine Solutions](#local-machine-solutions)
    - [Hosted Solutions](#hosted-solutions)
    - [Turn-key Cloud Solutions](#turn-key-cloud-solutions)
    - [Custom Solutions](#custom-solutions)
      - [Cloud](#cloud)
      - [On-Premises VMs](#on-premises-vms)
      - [Bare Metal](#bare-metal)
      - [Integrations](#integrations)
  - [Table of Solutions](#table-of-solutions)

<!-- END MUNGE: GENERATED_TOC -->


## Picking the Right Solution

If you just want to "kick the tires" on Kubernetes, we recommend the [local Docker-based](docker.html) solution.

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

- [Local Docker-based](docker.html) (recommended starting point)
- [Vagrant](vagrant.html) (works on any platform with Vagrant: Linux, MacOS, or Windows.)
- [No-VM local cluster](locally.html) (Linux only)


### Hosted Solutions

[Google Container Engine](https://cloud.google.com/container-engine) offers managed Kubernetes
clusters.

### Turn-key Cloud Solutions

These solutions allow you to create Kubernetes clusters on a range of Cloud IaaS providers with only a
few commands, and have active community support.

- [GCE](gce.html)
- [AWS](aws.html)
- [Azure](coreos/azure/README.html)

### Custom Solutions

Kubernetes can run on a wide range of Cloud providers and bare-metal environments, and with many
base operating systems.

If you can find a guide below that matches your needs, use it.  It may be a little out of date, but
it will be easier than starting from scratch.  If you do want to start from scratch because you
have special requirements or just because you want to understand what is underneath a Kubernetes
cluster, try the [Getting Started from Scratch](scratch.html) guide.

If you are interested in supporting Kubernetes on a new platform, check out our [advice for
writing a new solution](../../docs/devel/writing-a-getting-started-guide.html).

#### Cloud

These solutions are combinations of cloud provider and OS not covered by the above solutions.

- [AWS + coreos](coreos.html)
- [GCE + CoreOS](coreos.html)
- [AWS + Ubuntu](juju.html)
- [Joyent + Ubuntu](juju.html)
- [Rackspace + CoreOS](rackspace.html)

#### On-Premises VMs

- [Vagrant](coreos.html) (uses CoreOS and flannel)
- [CloudStack](cloudstack.html) (uses Ansible, CoreOS and flannel)
- [Vmware](vsphere.html)  (uses Debian)
- [juju.md](juju.html) (uses Juju, Ubuntu and flannel)
- [Vmware](coreos.html)  (uses CoreOS and flannel)
- [libvirt-coreos.md](libvirt-coreos.html)  (uses CoreOS)
- [oVirt](ovirt.html)
- [libvirt](fedora/flannel_multi_node_cluster.html) (uses Fedora and flannel)
- [KVM](fedora/flannel_multi_node_cluster.html)  (uses Fedora and flannel)

#### Bare Metal

- [Offline](coreos/bare_metal_offline.html) (no internet required.  Uses CoreOS and Flannel)
- [fedora/fedora_ansible_config.md](fedora/fedora_ansible_config.html)
- [Fedora single node](fedora/fedora_manual_config.html)
- [Fedora multi node](fedora/flannel_multi_node_cluster.html)
- [Centos](centos/centos_manual_config.html)
- [Ubuntu](ubuntu.html)
- [Docker Multi Node](docker-multinode.html)

#### Integrations

These solutions provide integration with 3rd party schedulers, resource managers, and/or lower level platforms.

- [Kubernetes on Mesos](mesos.html)
  - Instructions specify GCE, but are generic enough to be adapted to most existing Mesos clusters
- [Kubernetes on DCOS](dcos.html)
  - Community Edition DCOS uses AWS
  - Enterprise Edition DCOS supports cloud hosting, on-premise VMs, and bare metal

## Table of Solutions

Here are all the solutions mentioned above in table form.

IaaS Provider        | Config. Mgmt | OS     | Networking  | Docs                                              | Conforms | Support Level
-------------------- | ------------ | ------ | ----------  | ---------------------------------------------     | ---------| ----------------------------
GKE                  |              |        | GCE         | [docs](https://cloud.google.com/container-engine) | [✓][3]   | Commercial
Vagrant              | Saltstack    | Fedora | flannel     | [docs](vagrant.html)                                | [✓][2]   | Project
GCE                  | Saltstack    | Debian | GCE         | [docs](gce.html)                                    | [✓][1]   | Project
Azure                | CoreOS       | CoreOS | Weave       | [docs](coreos/azure/README.html)                    |          | Community ([@errordeveloper](https://github.com/errordeveloper), [@squillace](https://github.com/squillace), [@chanezon](https://github.com/chanezon), [@crossorigin](https://github.com/crossorigin))
Docker Single Node   | custom       | N/A    | local       | [docs](docker.html)                                 |          | Project ([@brendandburns](https://github.com/brendandburns))
Docker Multi Node    | Flannel      | N/A    | local       | [docs](docker-multinode.html)                       |          | Project ([@brendandburns](https://github.com/brendandburns))
Bare-metal           | Ansible      | Fedora | flannel     | [docs](fedora/fedora_ansible_config.html)           |          | Project
Digital Ocean        | custom       | Fedora | Calico      | [docs](fedora/fedora-calico.html)                   |          | Community (@djosborne)
Bare-metal           | custom       | Fedora | _none_      | [docs](fedora/fedora_manual_config.html)            |          | Project
Bare-metal           | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.html)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
libvirt              | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.html)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
KVM                  | custom       | Fedora | flannel     | [docs](fedora/flannel_multi_node_cluster.html)      |          | Community ([@aveshagarwal](https://github.com/aveshagarwal))
Mesos/Docker         | custom       | Ubuntu | Docker      | [docs](mesos-docker.html)                           |          | Community ([Kubernetes-Mesos Authors](https://github.com/mesosphere/kubernetes-mesos/blob/master/AUTHORS.md))
Mesos/GCE            |              |        |             | [docs](mesos.html)                                  |          | Community ([Kubernetes-Mesos Authors](https://github.com/mesosphere/kubernetes-mesos/blob/master/AUTHORS.md))
DCOS                 | Marathon   | CoreOS/Alpine | custom | [docs](dcos.html)                                   |          | Community ([Kubernetes-Mesos Authors](https://github.com/mesosphere/kubernetes-mesos/blob/master/AUTHORS.md))
AWS                  | CoreOS       | CoreOS | flannel     | [docs](coreos.html)                                 |          | Community
GCE                  | CoreOS       | CoreOS | flannel     | [docs](coreos.html)                                 |          | Community ([@pires](https://github.com/pires))
Vagrant              | CoreOS       | CoreOS | flannel     | [docs](coreos.html)                                 |          | Community ([@pires](https://github.com/pires), [@AntonioMeireles](https://github.com/AntonioMeireles))
Bare-metal (Offline) | CoreOS       | CoreOS | flannel     | [docs](coreos/bare_metal_offline.html)              |          | Community ([@jeffbean](https://github.com/jeffbean))
Bare-metal           | CoreOS       | CoreOS | Calico      | [docs](coreos/bare_metal_calico.html)               |          | Community ([@caseydavenport](https://github.com/caseydavenport))
CloudStack           | Ansible      | CoreOS | flannel     | [docs](cloudstack.html)                             |          | Community ([@runseb](https://github.com/runseb))
Vmware               |              | Debian | OVS         | [docs](vsphere.html)                                |          | Community ([@pietern](https://github.com/pietern))
Bare-metal           | custom       | CentOS | _none_      | [docs](centos/centos_manual_config.html)            |          | Community ([@coolsvap](https://github.com/coolsvap))
AWS                  | Juju         | Ubuntu | flannel     | [docs](juju.html)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
OpenStack/HPCloud    | Juju         | Ubuntu | flannel     | [docs](juju.html)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
Joyent               | Juju         | Ubuntu | flannel     | [docs](juju.html)                                   |          | [Community](https://github.com/whitmo/bundle-kubernetes) ( [@whit](https://github.com/whitmo), [@matt](https://github.com/mbruzek), [@chuck](https://github.com/chuckbutler) )
AWS                  | Saltstack    | Ubuntu | OVS         | [docs](aws.html)                                    |          | Community ([@justinsb](https://github.com/justinsb))
Bare-metal           | custom       | Ubuntu | Calico      | [docs](ubuntu-calico.html)                          |          | Community ([@djosborne](https://github.com/djosborne))
Bare-metal           | custom       | Ubuntu | flannel     | [docs](ubuntu.html)                                 |          | Community ([@resouer](https://github.com/resouer), [@WIZARD-CXY](https://github.com/WIZARD-CXY))
Local                |              |        | _none_      | [docs](locally.html)                                |          | Community ([@preillyme](https://github.com/preillyme))
libvirt/KVM          | CoreOS       | CoreOS | libvirt/KVM | [docs](libvirt-coreos.html)                         |          | Community ([@lhuard1A](https://github.com/lhuard1A))
oVirt                |              |        |             | [docs](ovirt.html)                                  |          | Community ([@simon3z](https://github.com/simon3z))
Rackspace            | CoreOS       | CoreOS | flannel     | [docs](rackspace.html)                              |          | Community ([@doublerr](https://github.com/doublerr))
any                  | any          | any    | any         | [docs](scratch.html)                                |          | Community ([@erictune](https://github.com/erictune))


*Note*: The above table is ordered by version test/used in notes followed by support level.

Definition of columns:

- **IaaS Provider** is who/what provides the virtual or physical machines (nodes) that Kubernetes runs on.
- **OS** is the base operating system of the nodes.
- **Config. Mgmt** is the configuration management system that helps install and maintain Kubernetes software on the
  nodes.
- **Networking** is what implements the [networking model](../../docs/admin/networking.html).  Those with networking type
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




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

