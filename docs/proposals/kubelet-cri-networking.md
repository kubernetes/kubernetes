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

# Container Runtime Interface (CRI) Networking Specifications

## Introduction
Container Runtime Interface (CRI) is an ongoing project to allow container 
runtimes to integrate with kubernetes via a newly-defined API. This document 
specifies the network requirements for container runtime 
interface (CRI). CRI networking requirements expand upon kubernetes pod 
networking requirements. This document does not specify requirements 
from upper layers of kubernetes network stack, such as `Service`. More 
background on k8s networking could be found 
[here](http://kubernetes.io/docs/admin/networking/)

## Requirements
1. Kubelet expects the runtime shim to manage pod’s network life cycle. Pod 
networking should be handled accordingly along with pod sandbox operations. 
   * `RunPodSandbox` must set up pod’s network. This includes, but is not limited 
to allocating a pod IP, configuring the pod’s network interfaces and default 
network route. Kubelet expects the pod sandbox to have an IP which is 
routable within the k8s cluster, if `RunPodSandbox` returns successfully. 
`RunPodSandbox` must return an error if it fails to set up the pod’s network. 
If the pod’s network has already been set up, `RunPodSandbox` must skip 
network setup and proceed. 
   * `StopPodSandbox` must tear down the pod’s network. The runtime shim 
must return error on network tear down failure. If pod’s network has 
already been torn down, `StopPodSandbox` must skip network tear down and proceed.
   * `RemovePodSandbox` may tear down pod’s network, if the networking has 
not been torn down already. `RemovePodSandbox` must return error on 
network tear down failure.
   * Response from `PodSandboxStatus` must include pod sandbox network status. 
The runtime shim must return an empty network status if it failed 
to construct a network status. 

2. User supplied pod networking configurations, which are NOT directly 
exposed by the kubernetes API, should be handled directly by runtime 
shims. For instance, `hairpin-mode`, `cni-bin-dir`, `cni-conf-dir`, `network-plugin`, 
`network-plugin-mtu` and `non-masquerade-cidr`. Kubelet will no longer handle 
these configurations after the transition to CRI is complete.
3. Network configurations that are exposed through the kubernetes API 
are communicated to the runtime shim through `UpdateRuntimeConfig` 
interface, e.g. `podCIDR`. For each runtime and network implementation, 
some configs may not be applicable. The runtime shim may handle or ignore 
network configuration updates from `UpdateRuntimeConfig` interface.

## Extensibility
* Kubelet is oblivious to how the runtime shim manages networking, i.e 
runtime shim is free to use [CNI](https://github.com/containernetworking/cni), 
[CNM](https://github.com/docker/libnetwork/blob/master/docs/design.md) or 
any other implementation as long as the CRI networking requirements and 
k8s networking requirements are satisfied.
* Runtime shims have full visibility into pod networking configurations. 
* As more network feature arrives, CRI will evolve. 

## Related Issues
* Kubelet network plugin for client/server container runtimes [#28667](https://github.com/kubernetes/kubernetes/issues/28667)
* CRI networking umbrella issue [#37316](https://github.com/kubernetes/kubernetes/issues/37316)