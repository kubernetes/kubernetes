<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Proposal: Flexible Kubernetes Cluster Creation with feature flags

> ***Please note: this proposal doesn't reflect final implementation, it's here for the purpose of capturing the original ideas.***
> ***You should probably [read `kubeadm` docs](http://kubernetes.io/docs/getting-started-guides/kubeadm/), to understand the end-result of this effor.***

Bogdan Dobrelya & many others in [SIG-cluster-lifecycle](https://github.com/kubernetes/community/tree/master/sig-cluster-lifecycle).

27th October 2016

*This proposal aims to capture the latest consensus and plan of action of SIG-cluster-lifecycle and the kubeadm dev team. It should satisfy the [feature request](https://github.com/kubernetes/features/issues/138).*

See also: [Kubeadm feature request](https://github.com/kubernetes/features/issues/11)

## Motivation

Kubernetes is hard to install. Kubeadm and numerous external installers solve that issue well. None of them are excellent, although together they might be a perfect team. We believe that flexibility that feature flags may bring to the former make this possible.

## Goals

Have a set of cluster bootstrapping tasks the kubeadm does as a list of configurable optional steps.
We plan to do so by introducing feature flags, either as config file options or CLI extentions to the `init` and `join` commands.

## Scope of the feature request

According the [Kubeadm feature request](https://github.com/kubernetes/features/issues/11) there are logically 3 steps to deploying a Kubernetes cluster:

1. *Provisioning*: Getting some servers - these may be VMs on a developer's workstation, VMs in public clouds, or bare-metal servers in a user's data center.

2. *Install & Discovery*: Installing the Kubernetes core components on those servers (kubelet, etc) - and bootstrapping the cluster to a state of basic liveness, including allowing each server in the cluster to discover other servers: for example teaching etcd servers about their peers, having TLS certificates provisioned, etc.

3. *Add-ons*: Now that basic cluster functionality is working, installing add-ons such as DNS or a pod network (should be possible using kubectl apply).

Feature flags shall allow users to execute olny given steps omitting the rest, if one wishes so. Integration with external CM solutions (aka installers) is out of the scope. This is only an enabler feature to make that integration possible in future.

The implementation of feature flags starts with the phase II of the kubeadm implementation.

## User stories

As a user I want to configure a list of tasks for the kubeadm init/join commands in the config file, like:

```
use_discovery: true
manage_certs: true
manage_pods: true
install_addons: true
external_etcd_endpoints: false
```

The given list of options may be not full.

Alternatively, I want to do as well via the extended `kubeadm init/join foo`
CLI, like:
```
master1$ kubeadm init create certs
master2$ kubeadm init sync certs <master1_ip>
node1$ kubadm join fetch certs <master1_ip>
master1$ kubeadm init create pods
master2$ kubeadm init create pods
node1$ kubeadm join create pods <master2_ip>
master2$ kubeadm init create discovery
node2$ kubeadm join
```
Here, the latter command assumes the node2 will do all required steps and autodiscover its
master automagically. While the node1 does only a partial join steps, which are
fetching certificates and creating the kube-proxy pod definition and can yet use
autodiscovery as it is yet configured. The master2 syncronizes certificates
from the master1, although this feature is not implemented and given just as en example.

Note that the config file may still contain some options and data required by some or all steps.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubeadm-bootstrap-feature-flags.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
