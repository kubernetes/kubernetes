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

Kubelet HyperContainer Container Runtime
=======================================

Authors: Pengfei Ni (@feiskyer), Harry Zhang (@resouer)

## Abstract

This proposal aims to support [HyperContainer](http://hypercontainer.io) container
runtime in Kubelet.

## Motivation

HyperContainer is a Hypervisor-agnostic Container Engine that allows you to run Docker images using
hypervisors (KVM, Xen, etc.). By running containers within separate VM instances, it offers a
hardware-enforced isolation, which is required in multi-tenant environments.

## Goals

1. Complete pod/container/image lifecycle management with HyperContainer.
2. Setup network by network plugins.
3. 100% Pass node e2e tests.
4. Easy to deploy for both local dev/test and production clusters.

## Design

The HyperContainer runtime will make use of the kubelet Container Runtime Interface. [Fakti](https://github.com/kubernetes/frakti) implements the CRI interface and exposes
a local endpoint to Kubelet. Fakti communicates with [hyperd](https://github.com/hyperhq/hyperd)
with its gRPC API to manage the lifecycle of sandboxes, containers and images.

![frakti](https://cloud.githubusercontent.com/assets/676637/18940978/6e3e5384-863f-11e6-9132-b638d862fd09.png)

## Limitations

Since pods are running directly inside hypervisor, host network is not supported in HyperContainer
runtime.

## Development

The HyperContainer runtime is maintained by <https://github.com/kubernetes/frakti>.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubelet-hypercontainer-runtime.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
