# Kubernetes Design Overview

Kubernetes build on top of [Docker](http://www.docker.io) to construct a clustered container scheduling service.  The goals of the project are to enable users to ask a Kubernetes cluster to run a set of containers.  The system will automatically pick a worker node to run those containers on.

As container based applications and systems get larger, some tools are provided to facilitate sanity. This includes ways for containers to find and communicate with each other and ways to work with and manage sets of containers that do similar work.

When looking at the arechitecture of the system, we'll break it down to services that run on the worker node and services that play a "master" role.

## Key Concept: Container Pod

While Docker itself works with individual containers, Kubernetes works with a `pod`.  A `pod` is a group of containers that are scheduled onto the same physical node.  In addition to defining the containers that run in the pod, the containers in the pod all use the same network namespace/IP and define a set of storage volumes.  Ports are also mapped on a per-pod basis.

## The Kubernetes Node

The Kubernetes node has the services necessary to run Docker containers and be managed from the master systems.

The Kubernetes node design is an extension of the [Container-optimized Google Compute Engine image](https://developers.google.com/compute/docs/containers#container-optimized_google_compute_engine_images).  Over time these plan is for these images/nodes to merge and be the same thing used in different ways. It has the services necessary to run Docker containers and be managed from the master systems.

Each node runs Docker, of course.  Docker takes care of the details of downloading images and running containers.

### Kubelet
The second component on the node called the `kubelet`.  The Kubelet is the logical successor (and rewrite in go) of the [Container Agent](https://github.com/GoogleCloudPlatform/container-agent) that is part of the Compute Engine image.

The Kubelet works in terms of a container manifest.  A container manifest (defined [here](https://developers.google.com/compute/docs/containers#container_manifest)) is a YAML file that describes a `pod`.  The Kubelet takes a set of manifests that are provided in various mechanisms and ensures that the containers described in those manifests are started and continue running.

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

It validates and configures data for 3 types of objects:

* `pod`: Each `pod` has a representation at the Kubernetes API level.
* `service`: A service is a configuration unit for the proxies that run on every worker node.  It is named and points to one or more Pods.
* `replicationController`: A replication controller takes a template and ensures that there is a specified number of "replicas" of that template running at any one time.  If there are too many, it'll start more.  If there are too few, it'll kill some.

Beyond just servicing REST operations, validating them and storing them in `etcd`, the API Server does two other things:

* Schedules pods to worker nodes.  Right now the scheduler is very simple.
* Synchronize pod information (where they are, what ports they are exposing) with the service configuration.

### Kubernetes Controller Manager Server

The `repliationController` type described above isn't strictly necessary for Kubernetes to be useful.  It is really a service that is layered on top of the simple `pod` API.  To enforce this layering, the logic for the repliationController is actually broken out into another server.  This server watches `etcd` for changes to `replicationController` objects and then uses the public Kubernetes API to implement the repliation algorithm.

### Key Concept: Labels

Pods are organized using labels.  Each pod can have a set of key/value labels set on it.

Via a "label query" the user can identify a set of `pods`.  This simple mechanism is a key part of how both `services` and `replicationControllers` work.  The set of pods that a `service` points at is defined with a label query.  Similarly the population of pods that a `replicationController` is monitoring is also defined with a label query.

Label queries would typically be used to identify and group pods into, say, a tier in an application.  You could also idenitfy the stack such as `dev`, `staging` or `production`.

These sets could be overlapping.  For instance, a service might point to all pods with `tier in (frontend), stack in (prod)`.  Now say you have 10 replicated pods that make up this tier.  But you want to be able to 'canary' a new version of this component.  You could set up a `replicationController` (with `replicas` set to 9) for the bulk of the replicas with labels `tier=frontend,stack=prod,canary=no` and another `replicationController` (with `replicas` set to 1) for the canary with labels `tier=frontend, stack=prod, canary=yes`.  Now the service is covering both the canary and non-canary pods.  But you can mess with the `replicationControllers` separately to test things out, monitor the results, etc.

## Network Model

Kubernetes expands the default Docker networking model.  The goal is to have each `pod` have an IP in a shared networking namespace that has full communication with other physical computers and containers across the network.  In this way, it becomes much less necessary to map ports.

For the Google Compute Engine cluster configuration scripts, [advanced routing](https://developers.google.com/compute/docs/networking#routing) is set up so that each VM has a extra 256 IP addresses that get routed to it.  This is in addition to the 'main' IP address assigned to the VM that is NAT-ed for Internet access.  The networking bridge (called `cbr0` to differentiate it from `docker0`) is set up outside of Docker proper and only does NAT for egress network traffic that isn't aimed at the virtual network.

Ports mapped in from the 'main IP' (and hence the internet if the right firewall rules are set up) are proxied in user mode by Docker.  In the future, this should be done with `iptables` by either the Kubelet or Docker: [Issue #15](https://github.com/GoogleCloudPlatform/kubernetes/issues/15).

## Release Process

Right now "building" or "releasing" Kubernetes consists of some scripts (in `release/` to create a `tar` of the necessary data and then uploading it to Google Cloud Storage.  In the future we will generate Docker images for the bulk of the above described components: [Issue #19](https://github.com/GoogleCloudPlatform/kubernetes/issues/19).

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

As there is no security currently built into the `apiserver`, the salt configuration will install `nginx`.  `nginx` is configured to serve HTTPS with a self signed certificate.  HTTP basic auth is used from the client to `nginx`.  `nginx` then forwards the request on to the `apiserver` over plain old HTTP.

The password is generated randomly as part of the `kube-up.sh` script and stored in `~/.kubernetes_auth`.
