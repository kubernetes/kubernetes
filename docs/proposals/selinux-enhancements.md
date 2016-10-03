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

## Abstract

Presents a proposal for enhancing the security of Kubernetes clusters using
SELinux and simplifying the implementation of SELinux support within the
Kubelet by removing the need to label the Kubelet directory with an SELinux
context usable from a container.

## Motivation

The current Kubernetes codebase relies upon the Kubelet directory being
labeled with an SELinux context usable from a container.  This means that a
container escaping namespace isolation will be able to use any file within the
Kubelet directory without defeating kernel MAC (mandatory access control).  In
order to limit the attack surface, we should enhance the Kubelet to relabel
any bind-mounts into containers into a usable SELinux context.

## Constraints and Assumptions

1.  No API changes allowed
2.  Behavior must be fully backward compatible
3.  No new admission controllers - make incremental improvements without huge
    refactorings

## Use Cases

1.  As a cluster operator, I want to avoid having to label the Kubelet
    directory with a label usable from a container, so that I can limit the
    attack surface available to a container escaping its namespace isolation
2.  As a user, I want to run a pod without an SELinux context explicitly
    specified and be isolated using MCS (multi-category security) on systems
    where SELinux is enabled, so that the pods on each host are isolated from
    one another
3.  As a user, I want to run a pod that uses the host IPC or PID namespace and
    want the system to do the right thing with regard to SELinux, so that no
    unnecessary relabel actions are performed

### Labeling the Kubelet directory

As previously stated, the current codebase relies on the Kubelet directory
being labeled with an SELinux context usable from a container.  The Kubelet
uses the SELinux context of this directory to determine what SELinux context
`tmpfs` mounts (provided by the EmptyDir memory-medium option) should receive.
The problem with this is that it opens an attack surface to a container that
escapes its namespace isolation.

### SELinux when no context is specified

When no SELinux context is specified, Kubernetes should just do the right
thing, where DTRT is defined as isolating pods with a node-unique set of
categories.  Node-uniqueness means unique among the pods scheduled onto the
node.  Long-term, we want to have a cluster-wide allocator for MCS labels.
Node-unique MCS labels are a good middle ground that is possible without a
new, large, feature.

### SELinux and host IPC and PID namespaces

Containers in pods that use the host IPC or PID namespaces are run with the
`spc_t` label, which means that the container process is effectively
unconfined by SELinux.  In this case, no relabeling is necessary.

## Analysis

### Libcontainer SELinux library

Docker and rkt both use the libcontainer SELinux library.  This library
provides a method, `GetLxcContexts`, that returns the correct base SELinux
contexts for container processes and files used by them.  Docker and rkt both
leverage this call to determine the 'starting' SELinux contexts for
containers.

### Docker

Docker's behavior when no SELinux context is defined is to pick a unique MCS
label for a container.  Additionally, a container B shares the IPC namespace
of another container A, container B is given the SELinux context of container
A.  What this means for Kubernetes pods (where containers share the IPC
namespace of the infra-container) is that in a vacuum the containers in a pod
should have the same SELinux context.

Docker is capable of relabeling bind-mounts into containers using the `:Z`
bind-mount flag.  However, in the current implementation of the docker runtime
in Kubernetes, the `:Z` option is only applied when the pod's SecurityContext
contains an SELinux context.  We could easily implement the correct behaviors
by always setting `:Z` on systems where SELinux is enabled.

In the case of a pod that shares the host IPC or PID
namespace, this flag is simply ignored and the container receives the `spc_t`
SELinux type.  The `spc_t` type is unconfined, and so no relabeling needs to
be done for volumes for these pods.  Currently, however, there is code which
relabels volumes into explicitly specified SELinux contexts for these pods.
This code is unnecessary and should be removed.

### rkt

TBD :)

## Proposed Changes

TBD

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/selinux-enhancements.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
