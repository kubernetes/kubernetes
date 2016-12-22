# Automated HA master deployment

**Author:** filipg@, jsz@

# Introduction

We want to allow users to easily replicate kubernetes masters to have highly available cluster,
initially using `kube-up.sh` and `kube-down.sh`.

This document describes technical design of this feature. It assumes that we are using aforementioned
scripts for cluster deployment. All of the ideas described in the following sections should be easy
to implement on GCE, AWS and other cloud providers.

It is a non-goal to design a specific setup for bare-metal environment, which
might be very different.

# Overview

In a cluster with replicated master, we will have N VMs, each running regular master components
such as apiserver, etcd, scheduler or controller manager. These components will interact in the
following way:
* All etcd replicas will be clustered together and will be using master election
  and quorum mechanism to agree on the state. All of these mechanisms are integral
  parts of etcd and we will only have to configure them properly.
* All apiserver replicas will be working independently talking to an etcd on
  127.0.0.1 (i.e. local etcd replica), which if needed will forward requests to the current etcd master
  (as explained [here](https://coreos.com/etcd/docs/latest/getting-started-with-etcd.html)).
* We will introduce provider specific solutions to load balance traffic between master replicas
  (see section `load balancing`)
* Controller manager, scheduler & cluster autoscaler will use lease mechanism and
  only a single instance will be an active master. All other will be waiting in a standby mode.
* All add-on managers will work independently and each of them will try to keep add-ons in sync

# Detailed design

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

All etcd instances will be clustered together and one of them will be an elected master.
In order to commit any change quorum of the cluster will have to confirm it. Etcd will be
configured in such a way that all writes and reads will go through the master (requests
will be forwarded by the local etcd server such that it’s invisible for the user). It will
affect latency for all operations, but it should not increase by much more than the network
latency between master replicas (latency between GCE zones with a region is < 10ms).

Currently etcd exposes port only using localhost interface. In order to allow clustering
and inter-VM communication we will also have to use public interface. To secure the
communication we will use SSL (as described [here](https://coreos.com/etcd/docs/latest/security.html)).

When generating command line for etcd we will always assume it’s part of a cluster
(initially of size 1) and list all existing kubernetes master replicas.
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

#### load balancing

With multiple apiservers we need a way to load balance traffic to/from master replicas. As different cloud
providers have different capabilities and limitations, we will not try to find a common lowest
denominator that will work everywhere. Instead we will document various options and apply different
solution for different deployments. Below we list possible approaches:

1. `Managed DNS` - user need to specify a domain name during cluster creation. DNS entries will be managed
automaticaly by the deployment tool that will be intergrated with solutions like Route53 (AWS)
or Google Cloud DNS (GCP). For load balancing we will have two options:
  1.1. create an L4 load balancer in front of all apiservers and update DNS name appropriately
  1.2. use round-robin DNS technique to access all apiservers directly
2. `Unmanaged DNS` - this is very similar to `Managed DNS`, with the exception that DNS entries
will be manually managed by the user. We will provide detailed documentation for the entries we
expect.
3. [GCP only] `Promote master IP` - in GCP, when we create the first master replica, we generate a static
external IP address that is later assigned to the master VM. When creating additional replicas we
will create a loadbalancer infront of them and reassign aforementioned IP to point to the load balancer
instead of a single master. When removing second to last replica we will reverse this operation (assign
IP address to the remaining master VM and delete load balancer). That way user will not have to provide
a domain name and all client configurations will keep working.

This will also impact `kubelet <-> master` communication as it should use load
balancing for it. Depending on the chosen method we will use it to properly configure
kubelet.

#### `kubernetes` service

Kubernetes maintains a special service called `kubernetes`. Currently it keeps a
list of IP addresses for all apiservers. As it uses a command line flag
`--apiserver-count` it is not very dynamic and would require restarting all
masters to change number of master replicas.

To allow dynamic changes to the number of apiservers in the cluster, we will
introduce a `ConfigMap` in `kube-system` namespace, that will keep an expiration
time for each apiserver (keyed by IP). Each apiserver will do three things:

1. periodically update expiration time for it's own IP address
2. remove all the stale IP addresses from the endpoints list
3. add it's own IP address if it's not on the list yet.

That way we will not only solve the problem of dynamically changing number
of apiservers in the cluster, but also the problem of non-responsive apiservers
that should be removed from the `kubernetes` service endpoints list.

#### Certificates

Certificate generation will work as today. In particular, on GCE, we will
generate it for the public IP used to access the cluster (see `load balancing`
section) and local IP of the master replica VM.

That means that with multiple master replicas and a load balancer in front
of them, accessing one of the replicas directly (using it's ephemeral public
IP) will not work on GCE without appropriate flags:

- `kubectl --insecure-skip-tls-verify=true`
- `curl --insecure`
- `wget --no-check-certificate`

For other deployment tools and providers the details of certificate generation
may be different, but it must be possible to access the cluster by using either
the main cluster endpoint (DNS name or IP address) or internal service called
`kubernetes` that points directly to the apiservers.

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

Command to add new replica on GCE using kube-up script:

```
KUBE_REPLICATE_EXISTING_MASTER=true KUBE_GCE_ZONE=us-central1-b kubernetes/cluster/kube-up.sh
```

A pseudo-code for adding a new master replica using managed DNS and a loadbalancer is the following:

```
1. If there is no load balancer for this cluster:
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
4. Add IP address of the new apiserver to the load balancer.
```

A simplified algorithm for adding a new master replica and promoting master IP to the load balancer
is identical to the one when using DNS, with a different step to setup load balancer:

```
1. If there is no load balancer for this cluster:
  1. Unassign IP from the existing master replica
  2. Create load balancer using static IP reclaimed in the previous step
  3. Add existing apiserver to the load balancer
  4. Wait until load balancer is working, i.e. all data is propagated, in GCE up to 20 min (sic!)
...
```

## Deleting replica

Command to delete one replica on GCE using kube-up script:

```
KUBE_DELETE_NODES=false KUBE_GCE_ZONE=us-central1-b kubernetes/cluster/kube-down.sh
```

A pseudo-code for deleting an existing replica for the master is the following:

```
1. Remove replica IP address from the load balancer or DNS configuration
2. SSH to one of the remaining masters and run the following command to remove replica from the cluster:
  `curl etcd-0:4001/v2/members/<id> -XDELETE -L`
3. Delete replica VM
4. If load balancer has only a single target instance, then delete load balancer
5. Update DNS to point to the remaining master replica, or [on GCE] assign static IP back to the master VM.
```

## Upgrades

Upgrading replicated master will be possible by upgrading them one by one using existing tools
(e.g. upgrade.sh for GCE). This will work out of the box because:
* Requests from nodes will be correctly served by either new or old master because apiserver is backward compatible.
* Requests from scheduler (and controllers) go to a local apiserver via localhost interface, so both components
will be in the same version.
* Apiserver talks only to a local etcd replica which will be in a compatible version
* We assume we will introduce this setup after we upgrade to etcd v3 so we don't need to cover upgrading database.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/ha_master.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
