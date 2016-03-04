<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Cluster Admin Guide

The cluster admin guide is for anyone creating or administering a Kubernetes cluster.
It assumes some familiarity with concepts in the [User Guide](../user-guide/README.md).

## Planning a cluster

There are many different examples of how to setup a kubernetes cluster.  Many of them are listed in this
[matrix](../getting-started-guides/README.md).  We call each of the combinations in this matrix a *distro*.

Before choosing a particular guide, here are some things to consider:

 - Are you just looking to try out Kubernetes on your laptop, or build a high-availability many-node cluster? Both
   models are supported, but some distros are better for one case or the other.
 - Will you be using a hosted Kubernetes cluster, such as [GKE](https://cloud.google.com/container-engine), or setting
   one up yourself?
 - Will your cluster be on-premises, or in the cloud (IaaS)?  Kubernetes does not directly support hybrid clusters.  We
   recommend setting up multiple clusters rather than spanning distant locations.
 - Will you be running Kubernetes on "bare metal" or virtual machines?  Kubernetes supports both, via different distros.
 - Do you just want to run a cluster, or do you expect to do active development of kubernetes project code?  If the
   latter, it is better to pick a distro actively used by other developers.  Some distros only use binary releases, but
   offer is a greater variety of choices.
 - Not all distros are maintained as actively.  Prefer ones which are listed as tested on a more recent version of
   Kubernetes.
 - If you are configuring kubernetes on-premises, you will need to consider what [networking
   model](networking.md) fits best.
 - If you are designing for very high-availability, you may want [clusters in multiple zones](multi-cluster.md).
 - You may want to familiarize yourself with the various
   [components](cluster-components.md) needed to run a cluster.

## Setting up a cluster

Pick one of the Getting Started Guides from the [matrix](../getting-started-guides/README.md) and follow it.
If none of the Getting Started Guides fits, you may want to pull ideas from several of the guides.

One option for custom networking is *OpenVSwitch GRE/VxLAN networking* ([ovs-networking.md](ovs-networking.md)), which
uses OpenVSwitch to set up networking between pods across
  Kubernetes nodes.

If you are modifying an existing guide which uses Salt, this document explains [how Salt is used in the Kubernetes
project](salt.md).

## Managing a cluster, including upgrades

[Managing a cluster](cluster-management.md).

## Managing nodes

[Managing nodes](node.md).

## Optional Cluster Services

* **DNS Integration with SkyDNS** ([dns.md](dns.md)):
  Resolving a DNS name directly to a Kubernetes service.

* **Logging** with [Kibana](../user-guide/logging.md)

## Multi-tenant support

* **Resource Quota** ([resource-quota.md](resource-quota.md))

## Security

* **Kubernetes Container Environment** ([docs/user-guide/container-environment.md](../user-guide/container-environment.md)):
  Describes the environment for Kubelet managed containers on a Kubernetes
  node.

* **Securing access to the API Server** [accessing the api](accessing-the-api.md)

* **Authentication**  [authentication](authentication.md)

* **Authorization** [authorization](authorization.md)

* **Admission Controllers** [admission_controllers](admission-controllers.md)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/introduction.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
