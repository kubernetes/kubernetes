# Kubernetes Design Overview

- [Overview](#overview)
- [Key Concepts](#key-concepts)
  - [Pods](#pods)
  - [Labels](#labels)
- [The Kubernetes Node](#the-kubernetes-node)
  - [Kubelet](#kubelet)
  - [Kubernetes Proxy](#kubernetes-proxy)
- [The Kubernetes Master](#the-kubernetes-master)
  - [etcd](#etcd)
  - [Kubernetes API Server](#kubernetes-api-server)
  - [Kubernetes Controller Manager Server](#kubernetes-controller-manager-server)
- [Network Model](#network-model)
- [Release Process](#release-process)
- [GCE Cluster Configuration](#gce-cluster-configuration)
  - [Cluster Security](#cluster-security)

## Overview

Kubernetes builds on top of [Docker](http://www.docker.io) to construct a clustered container scheduling service.  Kubernetes enables users to ask a cluster to run a set of containers.  The system will automatically pick worker nodes to run those containers on, which we think of more as "scheduling" than "orchestration". Kubernetes also provides ways for containers to find and communicate with each other and ways to manage both tightly coupled and loosely coupled sets of cooperating containers.

## Key Concepts

While Docker itself works with individual containers, Kubernetes provides higher-level organizational constructs in support of common cluster-level usage patterns, currently focused on service applications.

### Pods

A `pod` (as in a pod of whales or pea pod) is a relatively tightly coupled group of containers that are scheduled onto the same physical node.  In addition to defining the containers that run in the pod, the containers in the pod all use the same network namespace/IP (and port space), and define a set of shared storage volumes. Pods facilitate data sharing and IPC among their constituents, serve as units of scheduling, deployment, and horizontal scaling/replication, and share fate. In the future, they may share resources ([LPC2013](http://www.linuxplumbersconf.org/2013/ocw//system/presentations/1239/original/lmctfy%20(1).pdf)).

While pods can be used for vertical integration of application stacks, their primary motivation is to support co-located, co-managed helper programs, such as:
- content management systems, file and data loaders, local cache managers, etc.
- log and checkpoint backup, compression, rotation, snapshotting, etc.
- data change watchers, log tailers, logging and monitoring adapters, event publishers, etc.
- proxies, bridges, and adapters
- controllers, managers, configurators, and updaters

### Labels

Loosely coupled cooperating pods are organized using key/value labels.

Each pod can have a set of key/value labels set on it, with at most one label with a particular key. 

Individual labels are used to specify identifying metadata, and to convey the semantic purposes/roles of pods of containers. Examples of typical pod label keys include `environment` (e.g., with values `dev`, `qa`, or `production`), `service`, `tier` (e.g., with values `frontend` or `backend`), `partition`, and `track` (e.g., with values `daily` or `weekly`), but you are free to develop your own conventions.

Via a "label selector" the user can identify a set of `pods`. The label selector is the core grouping primitive in Kubernetes. It could be used to identify service replicas or shards, worker pool members, or peers in a distributed application.

Kubernetes currently supports two objects that use label selectors to keep track of their members, `service`s and `replicationController`s:
- `service`: A service is a configuration unit for the proxies that run on every worker node.  It is named and points to one or more Pods.
- `replicationController`: A replication controller takes a template and ensures that there is a specified number of "replicas" of that template running at any one time.  If there are too many, it'll kill some.  If there are too few, it'll start more.

The set of pods that a `service` targets is defined with a label selector. Similarly, the population of pods that a `replicationController` is monitoring is also defined with a label selector. 

Pods may be removed from these sets by changing their labels. This flexibility may be used to remove pods from service for debugging, data recovery, etc.

For management convenience and consistency, `services` and `replicationControllers` may themselves have labels and would generally carry the labels their corresponding pods have in common.

Sets identified by labels and label selectors could be overlapping (think Venn diagrams). For instance, a service might point to all pods with `tier in (frontend), environment in (prod)`.  Now say you have 10 replicated pods that make up this tier.  But you want to be able to 'canary' a new version of this component.  You could set up a `replicationController` (with `replicas` set to 9) for the bulk of the replicas with labels `tier=frontend, environment=prod, track=stable` and another `replicationController` (with `replicas` set to 1) for the canary with labels `tier=frontend, environment=prod, track=canary`.  Now the service is covering both the canary and non-canary pods.  But you can mess with the `replicationControllers` separately to test things out, monitor the results, etc. 

Note that the superset described in the previous example is also heterogeneous. In long-lived, highly available, horizontally scaled, distributed, continuously evolving service applications, heterogeneity is inevitable, due to canaries, incremental rollouts, live reconfiguration, simultaneous updates and auto-scaling, hardware upgrades, and so on.

Pods may belong to multiple sets simultaneously, which enables representation of service substructure and/or superstructure. In particular, labels are intended to facilitate the creation of non-hierarchical, multi-dimensional deployment structures. They are useful for a variety of management purposes (e.g., configuration, deployment) and for application introspection and analysis (e.g., logging, monitoring, alerting, analytics). Without the ability to form sets by intersecting labels, many implicitly related, overlapping flat sets would need to be created, for each subset and/or superset desired, which would lose semantic information and be difficult to keep consistent. Purely hierarchically nested sets wouldn't readily support slicing sets across different dimensions.

Since labels can be set at pod creation time, no separate set add/remove operations are necessary, which makes them easier to use than manual set management. Additionally, since labels are directly attached to pods and label selectors are fairly simple, it's easy for users and for clients and tools to determine what sets they belong to. OTOH, with sets formed by just explicitly enumerating members, one would (conceptually) need to search all sets to determine which ones a pod belonged to.

## The Kubernetes Node

When looking at the architecture of the system, we'll break it down to services that run on the worker node and services that play a "master" role.

The Kubernetes node has the services necessary to run Docker containers and be managed from the master systems.

The Kubernetes node design is an extension of the [Container-optimized Google Compute Engine image](https://developers.google.com/compute/docs/containers/container_vms).  Over time the plan is for these images/nodes to merge and be the same thing used in different ways. It has the services necessary to run Docker containers and be managed from the master systems.

Each node runs Docker, of course.  Docker takes care of the details of downloading images and running containers.

### Kubelet
The second component on the node is called the `kubelet`.  The Kubelet is the logical successor (and rewritten in go) of the [Container Agent](https://github.com/GoogleCloudPlatform/container-agent) that is part of the Compute Engine image.

The Kubelet works in terms of a container manifest.  A container manifest (defined [here](https://developers.google.com/compute/docs/containers/container_vms#container_manifest)) is a YAML file that describes a `pod`.  The Kubelet takes a set of manifests that are provided in various mechanisms and ensures that the containers described in those manifests are started and continue running.

There are 4 ways that a container manifest can be provided to the Kubelet:

* **File** Path passed as a flag on the command line.  This file is rechecked every 20 seconds (configurable with a flag).
* **HTTP endpoint** HTTP endpoint passed as a parameter on the command line.  This endpoint is checked every 20 seconds (also configurable with a flag.)
* **etcd server**  The Kubelet will reach out and do a `watch` on an [etcd](https://github.com/coreos/etcd) server.  The etcd path that is watched is `/registry/hosts/$(hostname -f)`.  As this is a watch, changes are noticed and acted upon very quickly.
* **HTTP server** The kubelet can also listen for HTTP and respond to a simple API (underspec'd currently) to submit a new manifest.

### Kubernetes Proxy

Each node also runs a simple network proxy.  This reflects `services` as defined in the Kubernetes API on each node and can do simple TCP stream forwarding or round robin TCP forwarding across a set of backends.

## The Kubernetes Master

The Kubernetes master is split into a set of components.  These work together to provide an unified view of the cluster.

### etcd

All persistent master state is stored in an instance of `etcd`.  This provides a great way to store configuration data reliably.  With `watch` support, coordinating components can be notified very quickly of changes.

### Kubernetes API Server

This server serves up the main [Kubernetes API](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/api).

It validates and configures data for 3 types of objects: `pod`s, `service`s, and `replicationController`s.

Beyond just servicing REST operations, validating them and storing them in `etcd`, the API Server does two other things:

* Schedules pods to worker nodes.  Right now the scheduler is very simple.
* Synchronize pod information (where they are, what ports they are exposing) with the service configuration.

### Kubernetes Controller Manager Server

The `replicationController` type described above isn't strictly necessary for Kubernetes to be useful.  It is really a service that is layered on top of the simple `pod` API.  To enforce this layering, the logic for the replicationController is actually broken out into another server.  This server watches `etcd` for changes to `replicationController` objects and then uses the public Kubernetes API to implement the replication algorithm.

## Network Model

Kubernetes expands the default Docker networking model.  The goal is to have each `pod` have an IP in a shared networking namespace that has full communication with other physical computers and containers across the network.  In this way, it becomes much less necessary to map ports.

For the Google Compute Engine cluster configuration scripts, [advanced routing](https://developers.google.com/compute/docs/networking#routing) is set up so that each VM has a extra 256 IP addresses that get routed to it.  This is in addition to the 'main' IP address assigned to the VM that is NAT-ed for Internet access.  The networking bridge (called `cbr0` to differentiate it from `docker0`) is set up outside of Docker proper and only does NAT for egress network traffic that isn't aimed at the virtual network.

Ports mapped in from the 'main IP' (and hence the internet if the right firewall rules are set up) are proxied in user mode by Docker.  In the future, this should be done with `iptables` by either the Kubelet or Docker: [Issue #15](https://github.com/GoogleCloudPlatform/kubernetes/issues/15).

## Release Process

Right now "building" or "releasing" Kubernetes consists of some scripts (in `release/`) to create a `tar` of the necessary data and then uploading it to Google Cloud Storage.  In the future we will generate Docker images for the bulk of the above described components: [Issue #19](https://github.com/GoogleCloudPlatform/kubernetes/issues/19).

## GCE Cluster Configuration

The scripts and data in the `cluster/` directory automates creating a set of Google Compute Engine VMs and installing all of the Kubernetes components.  There is a single master node and a set of worker (called minion) nodes.

`config-default.sh` has a set of tweakable definitions/parameters for the cluster.

The heavy lifting of configuring the VMs is done by [SaltStack](http://www.saltstack.com/).

The bootstrapping works like this:

1. The `kube-up.sh` script uses the GCE [`startup-script`](https://developers.google.com/compute/docs/howtos/startupscript) mechanism for both the master node and the minion nodes.
  * For the minion, this simply configures and installs SaltStack.  The network range that this minion is assigned is baked into the startup-script for that minion.
  * For the master, the release files are downloaded from GCS and unpacked.  Various parts (specifically the SaltStack configuration) are installed in the right places.
2. SaltStack then installs the necessary servers on each node.
  * All go code is currently downloaded to each machine and compiled at install time.
  * The custom networking bridge is configured on each minion before Docker is installed.
  * Configuration (like telling the `apiserver` the hostnames of the minions) is dynamically created during the saltstack install.
3. After the VMs are started, the `kube-up.sh` script will call `curl` every 2 seconds until the `apiserver` starts responding.

`kube-down.sh` can be used to tear the entire cluster down.  If you build a new release and want to update your cluster, you can use `kube-push.sh` to update and apply (`highstate` in salt parlance) the salt config.

### Cluster Security

As there is no security currently built into the `apiserver`, the salt configuration will install `nginx`.  `nginx` is configured to serve HTTPS with a self signed certificate.  HTTP basic auth is used from the client to `nginx`.  `nginx` then forwards the request on to the `apiserver` over plain old HTTP.  Because a self signed certificate is used, access to the server should be safe from eavesdropping but is subject to "man in the middle" attacks.  Access via the browser will result in warnings and tools like curl will require an "--insecure" flag.

All communication within the cluster (worker nodes to the master, for instance) occurs on the internal virtual network and should be safe from eavesdropping.

The password is generated randomly as part of the `kube-up.sh` script and stored in `~/.kubernetes_auth`.
