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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Proposal: Self-hosted kubelet

## Abstract

In a self-hosted Kubernetes deployment, we have the initial bootstrap problem. This proposal presents a solution to the kubelet bootstrap, and assumes a functioning control plane, and a kubelet that can securely contact the API server.

## Background

Our current approach to a self-hosted kubelet is a "pivot" style installation. This procedure assumes a short-lived “bootstrap” kubelet will run and start a long-running “self-hosted” kubelet. Once the self-hosted kubelet is running the bootstrap kubelet will exit. As part of this, we propose introducing a new `--bootstrap` flag to the kubelet. The behaviour of that flag will be explained in detail below.

## Proposal

We propose adding a new flag to the kubelet, the `--bootstrap` flag, which is assumed to be used in conjunction with the `--lock-file` flag. When the `--bootstrap` flag is provided, after the kubelet acquires the file lock, it will begin asynchronously waiting on [inotify](http://man7.org/linux/man-pages/man7/inotify.7.html) events. Once an "open" event is received, the kubelet will assume another kubelet is attempting to take control and will exit. In this process, the $init system would be responsible for ensuring a kubelet is always running on the node.

Thus, the initial bootstrap becomes:

1. "bootstrap" kubelet is started by $init system.
1. "bootstrap" kubelet pulls down "self-hosted" kubelet as a pod from a daemonset
1. "self-hosted" kubelet attempts to acquire the file lock, causing "bootstrap" kubelet to exit
1. "self-hosted" kubelet acquires lock and takes over
1. "bootstrap" kubelet is restarted by $init system and blocks on acquiring the file lock

During an upgrade of the kubelet, for simplicity we will consider 3 kubelets, namely "bootstrap", "v1", and "v2". We imagine the following scenario for upgrades:

1. Cluster administrator introduces "v2" kubelet daemonset
1. "v1" kubelet pulls down and starts "v2"
1. Cluster administrator removes "v1" kubelet daemonset
1. "v1" kubelet is killed
1. Both "bootstrap" and "v2" kubelets race for file lock
1. If "v2" kubelet acquires lock, process has completed
1. If "bootstrap" kubelet acquires lock, it is assumed that "v2" kubelet will fail a health check and be killed. Once restarted, it will try to acquire the lock, triggering the "bootstrap" kubelet to exit.

Alternatively, it would also be possible via this mechanism to delete the "v1" daemonset first, allow the "bootstrap" kubelet to take over, and then introduce the "v2" kubelet daemonset, effectively eliminating the race between "bootstrap" and "v2" for lock acquisition, and the reliance on the failing health check procedure.

This will allow a "self-hosted" kubelet with minimal new concepts introduced into the core Kubernetes code base, and remains flexible enough to work well with future [bootstrapping services](https://github.com/kubernetes/kubernetes/issues/5754).

## Other discussion

Various similar approaches have been discussed [here](https://github.com/kubernetes/kubernetes/issues/246#issuecomment-64533959) and [here](https://github.com/kubernetes/kubernetes/issues/23073#issuecomment-198478997). This also relies on the kubelet being able to be run inside a container, more discussion on that is [here](https://github.com/kubernetes/kubernetes/issues/4869).

Additionally, [Taints and Tolerations](../../docs/design/taint-toleration-dedicated.md), whose design has already been accepted, would make the overall kubelet bootstrap more deterministic. With this, we would also need the ability for a kubelet to register itself with a given taint when it first contacts the API server. Given that, a kubelet could register itself with a given taint such as “component=kubelet”, and a kubelet pod could exist that has a toleration to that taint, ensuring it is the only pod the “bootstrap” kubelet runs.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/self-hosted-kubelet.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
