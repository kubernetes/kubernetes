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

# Automated HA master deployment (in GCE)

**Author:** filipg@, jsz@

# Introduction

We want to allow users to easily replicate kubernetes masters to have highly available cluster,
initially using `kube-up.sh` and `kube-down.sh`.

This document describes technical design of this feature. It assumes that we are using aforementioned
scripts for cluster deployment. It focuses on GCE use-case but all of the ideas described in
the following sections should be easy to port to AWS and other cloud providers.

It is a non-goal to design a specific setup for bare-metal environment, which
might be very different.

# Overview

In a cluster with replicated master, we will have N VMs, each running regular master components
such as apiserver, etcd, scheduler or controller manager. These components will interact in the
following way:
* All etcd replicas will be clustered together and will be using master election
  and quorum mechanism to agree on the state. All of these mechanisms are integral
  parts of etcd and we will only have to configure them properly.
* All apiserver replicas will be working independently talking to a etcd on
  127.0.0.1 (i.e. local etcd replica), which if needed will forward requests to the current etcd master
  (as explained [here](https://coreos.com/etcd/docs/latest/getting-started-with-etcd.html)).
* We will introduce a DNS name for a cluster and optionally add a load balancer
  in front of apiservers to spread the load among all the replicas.
* Controller manager, scheduler & cluster autoscaler will use lease mechanism and
  only a single instance will be an active master. All other will be waiting in a standby mode.
* All add-on managers will work independently and each of them will try to keep add-ons in sync

# Detailed design

## VM naming in GCE

VMs with master replicas will have the following names:
* first VM - `${MASTER_NAME}`
* next VMs - `${MASTER_NAME}-<hash>`, where `<hash>` is a 3-characters long alphanumeric string.

The "first VM" isn't special in any way and so it will be ok to delete it.

When adding more master replicas to the cluster (see section `Adding replica`), naming pattern
need to stay unchanged, i.e. `${MASTER_NAME}` must stay the same for all replicas.

## Components

### etcd

```
Note: This design for etcd clustering is quite pet-set like - each etcd
replica has its name which is explicitly used in etcd configuration etc. In
medium-term future we would like to have the ability to run masters as part of
autoscaling-group (AWS) or managed-instance-group (GCE) and add/remove replicas
automatically. This is pretty tricky and this design does not cover this.
It will be covered in a separate doc.
```

```
Note: This section assumes we are using etcd v2.2; it will have to be revisited during upgrade
to v3, e.g. different command line options.
```

All etcd instances will be clustered together and one of them will be an elected master.
In order to commit any change quorum of the cluster will have to confirm it. Etcd will be
configured in such a way that all writes and reads will go through the master (requests
will be forwarded by the local etcd server such that it’s invisible for the user). It will
affect latency for all operations, but it should not increase by much more than the network
latency between master replicas (latency between GCE zones with a region is < 10ms).

Currently etcd exposes port only using localhost interface. In order to allow clustering
and inter-VM communication we will also have to use public interface. To secure the
communication we will configure firewall rules to only allow traffic on etcd port from
other master machines.

Another option to secure etcd cluster communication would be to use SSL (as described
[here](https://coreos.com/etcd/docs/latest/security.html)), but it would increase CPU usage
on master machines, which is already a bottleneck in large clusters. In many
use cases it is not required so we will not offer this in the first version.

When generating command line for etcd we will always assume it’s part of a cluster
(initially of size 1) and list all existing kubernetes master replicas (see `VM naming in GCE` section).
Based on that, we will set the following flags:
* `-initial-cluster` - list of all hostnames/DNS names for master replicas (including the new one)
* `-initial-cluster-state` (keep in mind that we are adding master replicas one by one):
  * `new` if we are adding the first replica, i.e. the list of existing master replicas is empty
  * `existing` if there are more than one replica, i.e. the list of existing master replicas is non-empty.

This will allow us to have exactly the same logic for HA and non-HA master. List of DNS names for VMs
with master replicas will be generated in `kube-up.sh` script and passed to as a env variable
`INITIAL_ETCD_CLUSTER`.

### apiservers

All apiservers will work independently. They will contact etcd on 127.0.0.1, i.e. they will always contact
etcd replica running on the same VM. If needed, such requests will be forwarded by etcd server to the
etcd leader. This functionality is completely hidden from the client (apiserver
in our case).

Caching mechanism, which is implemented in apiserver, will not be affected by
replicating master because:
* GET requests go directly to etcd
* LIST requests go either directly to etcd or to cache populated via watch
  (depending on the ResourceVersion in ListOptions). In the second scenario,
  after a PUT/POST request, changes might not be visible in LIST response.
  This is however not worse than it is with the current single master.
* WATCH does not give any guarantees when change will be delivered.

#### DNS & load balancer

In order to expose a single endpoint for the user, we will introduce a public DNS name for the cluster.
For a non-replicated master it will point directly to the apiserver. For a replicated master
we will have two options:

* create an L4 load balancer in front of all apiservers and update DNS name appropriately
* use round-robin DNS technique to access all apiservers

The advantage from the client perspective of introducing a DNS name/load-balancer is that
it will be transparent how many apiservers we have.

#### Certificates

Certificates used by apiserver, will be generated for each instance separately in
`configure-vm.sh` script. Each of them will be valid for the IP address of a
given instance and a DNS name for the cluster. That way old clients (i.e. configured
before replicating the master) will not need to be changed, however they will still
use IP of the first master instance, not DNS name of the cluster.
New clients, using DNS name, will be able to contact all replicas. As a result,
if administrator adds more master replicas, old clients will not take advantage of this.
This fix this clients will have to *manually* change cluster address in their configuration.

To mitigate the need to manually update multiple clients we could also take the
initial IP for the master VM and make it a virtual one and point it to all the
replicas. This, however, will not be automated.

#### `kubernetes` service

Kubernetes maintains a special service called `kubernetes`. Currently it keeps a
list of IP addresses for all apiservers. As it uses a command line flag
`--apiserver-count` it is not very dynamic and would require restarting all
masters to change number of master replicas.

To allow dynamic changes to the number of apiservers in the cluster, we will introduce
a `ConfigMap` with cluster level configuration, that will keep it. Lack of this map
will mean we will use value from the command line flag.

Such cluster-level configuration object is needed also by other components, so
this `ConfigMap` should be generic enough to serve other needs as well (e.g.
network related info).

### controller manager, scheduler & cluster autoscaler

Controller manager and scheduler will by default use a lease mechanism to choose an active instance
among all masters. Only one instance will be performing any operations.
All other will be waiting in standby mode.

We will use the same configuration in non-replicated mode to simplify deployment scripts.

### add-on manager

All add-on managers will be working independently. Each of them will observe current state of
add-ons and will try to sync it with files on disk. As a result, due to races, a single add-on
can be updated multiple times in a row after upgrading the master. Long-term we should fix this
by using a similar mechanisms as controller manager or scheduler. However, currently add-on
manager is just a bash script and adding a master election mechanism would not be easy.

## Adding replica

Command to add new replica:

```
KUBE_REPLICATE_EXISTING_MASTER=true KUBE_GCE_ZONE=us-central1-b kubernetes/cluster/kube-up.sh
```

A simplified algorithm for adding a new replica for the master is the following:

```
1. [only if using load balancer] If there is no load balancer for this cluster:
  1. Create load balancer using ephemeral IP address
  2. Add existing apiserver to the load balancer
  3. Wait until load balancer is working, i.e. all data is propagated, in GCE up to 20 min (sic!)
  4. Update DNS to point to the load balancer.
2. Clone existing master (create a new VM with the same configuration) including
   all env variables (certificates, IP ranges etc), with the exception of
   `INITIAL_ETCD_CLUSTER`.
3. SSH to an existing master and run the following command to extend etcd cluster
   with the new instance:
   `curl <existing_master>:4001/v2/members -XPOST -H "Content-Type: application/json" -d '{"peerURLs":["http://<new_master>:2380"]}'`
4. Add IP address of the new apiserver to the load balancer or DNS.
```

## Deleting replica

Command to delete one replica:

```
KUBE_DELETE_NODES=false KUBE_GCE_ZONE=us-central1-b kubernetes/cluster/kube-down.sh
```

A simplified algorithm for deleting an existing replica for the master is the following:

```
1. Remove replica IP address from the load balancer or DNS configuration
2. SSH to one of the remaining masters and run the following command to remove replica from the cluster:
  `curl etcd-0:4001/v2/members/<id> -XDELETE -L`
3. Delete replica VM
4. If load balancer has only a single target instance, then delete load balancer
```

## Upgrades

Upgrading replicated master will be possible by upgrading them one by one using existing tools
(e.g. upgrade.sh for GCE). This will work out of the box because:
* Requests from nodes will be correctly served by either new or old master because apiserver is backward compatible.
* Requests from scheduler (and controllers) go to a local apiserver via localhost interface, so both components
will be in the same version.
* Apiserver talks only to a local etcd replica which will be in a compatible version
* Etcd replicas have backward compatible clustering & communication protocol so they will be able to share
state even when they are in different versions, including upgrade from `v2` to `v3`. The only problem which remains is with
upgrading storage format for etcd from v2 to v3, and hence upgrading APIs. This however is unrelated to
HA master and will be discussed in a separate document, specific to etcd v3 upgrade.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/ha_master.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
