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
[here](http://releases.k8s.io/release-1.0/docs/admin/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Cluster Admin Guide

The cluster admin guide is for anyone creating or administering a Kubernetes cluster.
It assumes some familiarity with concepts in the [User Guide](../user-guide/README.md).

## Admin Guide Table of Contents

[Introduction](introduction.md)

1. [Components of a cluster](cluster-components.md)
  1. [Cluster Management](cluster-management.md)
  1. Administrating Master Components
    1. [The kube-apiserver binary](kube-apiserver.md)
      1. [Authorization](authorization.md)
      1. [Authentication](authentication.md)
      1. [Accessing the api](accessing-the-api.md)
      1. [Admission Controllers](admission-controllers.md)
      1. [Administrating Service Accounts](service-accounts-admin.md)
      1. [Resource Quotas](resource-quota.md)
    1. [The kube-scheduler binary](kube-scheduler.md)
    1. [The kube-controller-manager binary](kube-controller-manager.md)
  1. [Administrating Kubernetes Nodes](node.md)
    1. [The kubelet binary](kubelet.md)
      1. [Garbage Collection](garbage-collection.md)
    1. [The kube-proxy binary](kube-proxy.md)
  1. Administrating Addons
    1. [DNS](dns.md)
  1. [Networking](networking.md)
    1. [OVS Networking](ovs-networking.md)
  1. Example Configurations
    1. [Multiple Clusters](multi-cluster.md)
    1. [High Availability Clusters](high-availability.md)
    1. [Large Clusters](cluster-large.md)
    1. [Getting started from scratch](../getting-started-guides/scratch.md)
      1. [Kubernetes's use of salt](salt.md)
  1. [Troubleshooting](cluster-troubleshooting.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
