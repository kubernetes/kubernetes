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

*This proposal aims to capture the latest consensus and plan of action of SIG-cluster-lifecycle and kubeadm dev team.
It shall be amended to the original [feature request](https://github.com/kubernetes/features/issues/11) once its design accepted.*

## Motivation

Kubernetes is hard to install. Kubeadm and numerous external installers solve that issue well. None of them are excellent, although
together they might be a perfect team. We believe that flexibility that feature flags may bring to the former make this possible.

The main use case is: as a user of kubernetes ansible/salt/puppet/name-it cluster installer, I want to make possible a step by step
integration of kubeadm into deployment playbooks/manifests/formulas/etc. I can't/don’t want do that just all at once and by running
`kubeadm init` or `kubadm join`, I need some deployment steps to remain as is, some delegated to kubeadm, with a hope of deployments
to be done fully by kubeadm with only some minor CM tasks remained as playbooks/manifests, eventually.

## Goals

Ease integration of kubeadm with external configuration management. We plan to do so by introducing opt-in feature flags to the
`init` and `join` commands. Integration of kubeadm with configuration management tools may not be done as instant, some tasks should
be delegated earlier, some tasks - later. Some tasks may require reordering or retrying as well. The opt-in feature flags to the rescue!

## Scope of the feature request

According the [Kubeadm feature request](https://github.com/kubernetes/features/issues/11) there are logically 3 steps to deploying a Kubernetes cluster:

1. *Provisioning*: Getting some servers - these may be VMs on a developer's workstation, VMs in public clouds, or bare-metal servers in a user's data center.

2. *Install & Discovery*: Installing the Kubernetes core components on those servers (kubelet, etc) - and bootstrapping the cluster to a state of basic liveness, including allowing each server in the cluster to discover other servers: for example teaching etcd servers about their peers, having TLS certificates provisioned, etc.

3. *Add-ons*: Now that basic cluster functionality is working, installing add-ons such as DNS or a pod network (should be possible using kubectl apply).

Feature flags shall allow users to execute olny given steps omitting the rest, if one wishes so. Integration with external CM solutions (aka installers) is out of the scope. This is only an enabler feature to make that integration possible in future.

The implementation of feature flags starts with the phase II of kubeadm implementation.

## Implementation plan

* Break out functionality into "phases". Make each of those phases a separate unit/package that gets registered.
* Make sure we track what info is needed for which phase and create config objects for that
* Change our logging output so that all output is marked with which phase it belongs to. This will help create the connection in the users' mind.
* Have `kubeadm init` and `kubeadm join` do be nothing more than running the separate phases. Present them that way in our documentation.
  * Bonus: Have `kubeadm init --plan` output the phases along with the corresponding `kubeadm manual <phase>` command.
  * Bonus: Have a `--skip-phase` flag that will skip a specific phase for init and join.
* Have `kubeadm manual <phase>` for each phase or functionality.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubeadm-bootstrap-feature-flags.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
