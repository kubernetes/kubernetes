<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Proposal: Self-hosted kubelet

## Abstract

In a self-hosted Kubernetes deployment (see [this
comment](https://github.com/kubernetes/kubernetes/issues/246#issuecomment-64533959)
for background on self hosted kubernetes), we have the initial bootstrap problem.
When running self-hosted components, there needs to be a mechanism for pivoting
from the initial bootstrap state to the kubernetes-managed (self-hosted) state.
In the case of a self-hosted kubelet, this means pivoting from the initial
kubelet defined and run on the host, to the kubelet pod which has been scheduled
to the node.

This proposal presents a solution to the kubelet bootstrap, and assumes a
functioning control plane (e.g. an apiserver, controller-manager, scheduler, and
etcd cluster), and a kubelet that can securely contact the API server. This
functioning control plane can be temporary, and not necessarily the "production"
control plane that will be used after the initial pivot / bootstrap.

## Background and Motivation

In order to understand the goals of this proposal, one must understand what
"self-hosted" means. This proposal defines "self-hosted" as a kubernetes cluster
that is installed and managed by the kubernetes installation itself. This means
that each kubernetes component is described by a kubernetes manifest (Daemonset,
Deployment, etc) and can be updated via kubernetes.

The overall goal of this proposal is to make kubernetes easier to install and
upgrade. We can then treat kubernetes itself just like any other application
hosted in a kubernetes cluster, and have access to easy upgrades, monitoring,
and durability for core kubernetes components themselves.

We intend to achieve this by using kubernetes to manage itself.  However, in
order to do that we must first "bootstrap" the cluster, by using kubernetes to
install kubernetes components. This is where this proposal fits in, by
describing the necessary modifications, and required procedures, needed to run a
self-hosted kubelet.

The approach being proposed for a self-hosted kubelet is a "pivot" style
installation.  This procedure assumes a short-lived “bootstrap” kubelet will run
and start a long-running “self-hosted” kubelet. Once the self-hosted kubelet is
running the bootstrap kubelet will exit. As part of this, we propose introducing
a new `--bootstrap` flag to the kubelet. The behaviour of that flag will be
explained in detail below.

## Proposal

We propose adding a new flag to the kubelet, the `--bootstrap` flag, which is
assumed to be used in conjunction with the `--lock-file` flag. The `--lock-file`
flag is used to ensure only a single kubelet is running at any given time during
this pivot process. When the `--bootstrap` flag is provided, after the kubelet
acquires the file lock, it will begin asynchronously waiting on
[inotify](http://man7.org/linux/man-pages/man7/inotify.7.html) events. Once an
"open" event is received, the kubelet will assume another kubelet is attempting
to take control and will exit by calling `exit(0)`.

Thus, the initial bootstrap becomes:

1. "bootstrap" kubelet is started by $init system.
1. "bootstrap" kubelet pulls down "self-hosted" kubelet as a pod from a
   daemonset
1. "self-hosted" kubelet attempts to acquire the file lock, causing "bootstrap"
   kubelet to exit
1. "self-hosted" kubelet acquires lock and takes over
1. "bootstrap" kubelet is restarted by $init system and blocks on acquiring the
   file lock

During an upgrade of the kubelet, for simplicity we will consider 3 kubelets,
namely "bootstrap", "v1", and "v2". We imagine the following scenario for
upgrades:

1. Cluster administrator introduces "v2" kubelet daemonset
1. "v1" kubelet pulls down and starts "v2"
1. Cluster administrator removes "v1" kubelet daemonset
1. "v1" kubelet is killed
1. Both "bootstrap" and "v2" kubelets race for file lock
1. If "v2" kubelet acquires lock, process has completed
1. If "bootstrap" kubelet acquires lock, it is assumed that "v2" kubelet will
   fail a health check and be killed. Once restarted, it will try to acquire the
   lock, triggering the "bootstrap" kubelet to exit.

Alternatively, it would also be possible via this mechanism to delete the "v1"
daemonset first, allow the "bootstrap" kubelet to take over, and then introduce
the "v2" kubelet daemonset, effectively eliminating the race between "bootstrap"
and "v2" for lock acquisition, and the reliance on the failing health check
procedure.

Eventually this could be handled by a DaemonSet upgrade policy.

This will allow a "self-hosted" kubelet with minimal new concepts introduced
into the core Kubernetes code base, and remains flexible enough to work well
with future [bootstrapping
services](https://github.com/kubernetes/kubernetes/issues/5754).

## Production readiness considerations / Out of scope issues

* Deterministically pulling and running kubelet pod: we would prefer not to have
  to loop until we finally get a kubelet pod.
* It is possible that the bootstrap kubelet version is incompatible with the
  newer versions that were run in the node. For example, the cgroup
  configurations might be incompatible. In the beginning, we will require
  cluster admins to keep the configuration in sync. Since we want the bootstrap
  kubelet to come up and run even if the API server is not available, we should
  persist the configuration for bootstrap kubelet on the node. Once we have
  checkpointing in kubelet, we will checkpoint the updated config and have the
  bootstrap kubelet use the updated config, if it were to take over.
* Currently best practice when upgrading the kubelet on a node is to drain all
  pods first. Automatically draining of the node during kubelet upgrade is out
  of scope for this proposal. It is assumed that either the cluster
  administrator or the daemonset upgrade policy will handle this.

## Other discussion

Various similar approaches have been discussed
[here](https://github.com/kubernetes/kubernetes/issues/246#issuecomment-64533959)
and
[here](https://github.com/kubernetes/kubernetes/issues/23073#issuecomment-198478997).
Other discussion around the kubelet being able to be run inside a container is
[here](https://github.com/kubernetes/kubernetes/issues/4869). Note this isn't a
strict requirement as the kubelet could be run in a chroot jail via rkt fly or
other such similar approach.

Additionally, [Taints and
Tolerations](../../docs/design/taint-toleration-dedicated.md), whose design has
already been accepted, would make the overall kubelet bootstrap more
deterministic. With this, we would also need the ability for a kubelet to
register itself with a given taint when it first contacts the API server. Given
that, a kubelet could register itself with a given taint such as
“component=kubelet”, and a kubelet pod could exist that has a toleration to that
taint, ensuring it is the only pod the “bootstrap” kubelet runs.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/self-hosted-kubelet.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
