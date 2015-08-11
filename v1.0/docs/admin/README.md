---
layout: docwithnav
title: "Kubernetes Cluster Admin Guide"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Cluster Admin Guide

The cluster admin guide is for anyone creating or administering a Kubernetes cluster.
It assumes some familiarity with concepts in the [User Guide](../user-guide/README.html).

## Admin Guide Table of Contents

[Introduction](introduction.html)

1. [Components of a cluster](cluster-components.html)
  1. [Cluster Management](cluster-management.html)
  1. Administrating Master Components
    1. [The kube-apiserver binary](kube-apiserver.html)
      1. [Authorization](authorization.html)
      1. [Authentication](authentication.html)
      1. [Accessing the api](accessing-the-api.html)
      1. [Admission Controllers](admission-controllers.html)
      1. [Administrating Service Accounts](service-accounts-admin.html)
      1. [Resource Quotas](resource-quota.html)
    1. [The kube-scheduler binary](kube-scheduler.html)
    1. [The kube-controller-manager binary](kube-controller-manager.html)
  1. [Administrating Kubernetes Nodes](node.html)
    1. [The kubelet binary](kubelet.html)
    1. [The kube-proxy binary](kube-proxy.html)
  1. Administrating Addons
    1. [DNS](dns.html)
  1. [Networking](networking.html)
    1. [OVS Networking](ovs-networking.html)
  1. Example Configurations
    1. [Multiple Clusters](multi-cluster.html)
    1. [High Availability Clusters](high-availability.html)
    1. [Large Clusters](cluster-large.html)
    1. [Getting started from scratch](../getting-started-guides/scratch.html)
      1. [Kubernetes's use of salt](salt.html)
  1. [Troubleshooting](cluster-troubleshooting.html)


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

