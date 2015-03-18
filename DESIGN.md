# Kubernetes Design Overview

- [Overview](#overview)
- [Key Concepts](#key-concepts)
  - [Pods](#pods)
  - [Labels](#labels)
- [The Kubernetes Node](#the-kubernetes-node)
  - [Kubelet](#kubelet)
  - [Kubernetes Proxy](#kubernetes-proxy)
- [The Kubernetes Control Plane](#the-kubernetes-control-plane)
  - [etcd](#etcd)
  - [Kubernetes API Server](#kubernetes-api-server)
  - [Kubernetes Controller Manager Server](#kubernetes-controller-manager-server)
- [GCE Cluster Configuration](#gce-cluster-configuration)
  - [Cluster Security](#cluster-security)

## Overview

Kubernetes is a system for managing containerized applications across multiple hosts, providing basic mechanisms for deployment, maintenance, and scaling of applications. Its APIs are intended to serve as the foundation for an open ecosystem of tools, automation systems, and higher-level API layers.

Kubernetes uses [Docker](http://www.docker.io) to package, instantiate, and run containerized applications.

Is Kubernetes, then, a Docker "orchestration" system? Yes and no.

Kubernetes establishes robust declarative primitives for maintaining the desired state requested by the user. We see these primitives as the main value added by Kubernetes. Self-healing mechanisms, such as auto-restarting, re-scheduling, and replicating containers require active controllers, not just imperative orchestration.

Kubernetes is primarily targeted at applications comprised of multiple containers, such as elastic, distributed micro-services. It is also designed to facilitate migration of non-containerized application stacks to Kubernetes. It therefore includes abstractions for grouping containers in both loosely coupled and tightly coupled formations, and provides ways for containers to find and communicate with each other in relatively familiar ways.

Kubernetes enables users to ask a cluster to run a set of containers. The system automatically chooses hosts to run those containers on. While Kubernetes's scheduler is currently very simple, we expect it to grow in sophistication over time. Scheduling is a policy-rich, topology-aware, workload-specific function that significantly impacts availability, performance, and capacity. The scheduler needs to take into account individual and collective resource requirements, quality of service requirements, hardware/software/policy constraints, affinity and anti-affinity specifications, data locality, inter-workload interference, deadlines, and so on. Workload-specific requirements will be exposed through the API as necessary.

Kubernetes is intended to run on a number of cloud providers, as well as on physical hosts.

A single Kubernetes cluster is not intended to span multiple availability zones. Instead, we recommend building a higher-level layer to replicate complete deployments of highly available applications across multiple zones (see [the availability doc](docs/availability.md) for more details).

Kubernetes is not currently suitable for use by multiple users -- see [Cluster Security](#cluster-security), below.

Finally, Kubernetes aspires to be an extensible, pluggable, building-block OSS platform and toolkit. Therefore, architecturally, we want Kubernetes to be built as a collection of pluggable components and layers, with the ability to use alternative schedulers, controllers, storage systems, and distribution mechanisms, and we're evolving its current code in that direction. Furthermore, we want others to be able to extend Kubernetes functionality, such as with higher-level PaaS functionality or multi-cluster layers, without modification of core Kubernetes source. Therefore, its API isn't just (or even necessarily mainly) targeted at end users, but at tool and extension developers. Consequently, there are no "internal" inter-component APIs. All APIs are visible and available, including the APIs used by the scheduler, the node controller, the replication-controller manager, Kubelet's API, etc. There's no glass to break -- in order to handle more complex use cases, one can just access the lower-level APIs in a fully transparent, composable manner.

### Cluster Architecture

A running Kubernetes cluster contains node agents (kubelet) and master components (APIs, scheduler, etc), on top of a distributed storage solution. This diagram shows our desired eventual state, though we're still working on a few things, like making kubelet itself (all our components, really) run within docker, and making the scheduler 100% pluggable.

![Architecture Diagram](/docs/architecture.png?raw=true "Architecture overview")

## Key Concepts

While Docker itself works with individual containers, Kubernetes provides higher-level organizational constructs in support of common cluster-level usage patterns, currently focused on service applications, but which could also be expanded to batch and test workloads in the future.

### Pods

A _pod_ (as in a pod of whales or pea pod) is a relatively tightly coupled group of containers that are scheduled onto the same host. It models an application-specific "virtual host" in a containerized environment. Pods serve as units of scheduling, deployment, and horizontal scaling/replication, share fate, and share some resources, such as storage volumes and IP addresses.

[More details on pods](docs/pods.md).

### Labels

Loosely coupled cooperating pods are organized using key/value _labels_.

Individual labels are used to specify identifying metadata, and to convey the semantic purposes/roles of pods of containers. Examples of typical pod label keys include `service`, `environment` (e.g., with values `dev`, `qa`, or `production`), `tier` (e.g., with values `frontend` or `backend`), and `track` (e.g., with values `daily` or `weekly`), but you are free to develop your own conventions.

Via a _label selector_ the user can identify a set of pods. The label selector is the core grouping primitive in Kubernetes. It could be used to identify service replicas or shards, worker pool members, or peers in a distributed application.

Kubernetes currently supports two objects that use label selectors to keep track of their members, `service`s and `replicationController`s:
- `service`: A service is a configuration unit for the [proxies](#kubernetes-proxy) that run on every worker node.  It is named and points to one or more pods.
- `replicationController`: A replication controller takes a template and ensures that there is a specified number of "replicas" of that template running at any one time.  If there are too many, it'll kill some.  If there are too few, it'll start more.

The set of pods that a `service` targets is defined with a label selector. Similarly, the population of pods that a `replicationController` is monitoring is also defined with a label selector.

For management convenience and consistency, `services` and `replicationControllers` may themselves have labels and would generally carry the labels their corresponding pods have in common.

[More details on labels](docs/labels.md).

## The Kubernetes Node

When looking at the architecture of the system, we'll break it down to services that run on the worker node and services that comprise the cluster-level control plane.

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

Each node also runs a simple network proxy.  This reflects `services` (see [here](docs/services.md) for more details) as defined in the Kubernetes API on each node and can do simple TCP and UDP stream forwarding (round robin) across a set of backends.

Service endpoints are currently found through environment variables (both [Docker-links-compatible](https://docs.docker.com/userguide/dockerlinks/) and Kubernetes {FOO}_SERVICE_HOST and {FOO}_SERVICE_PORT variables are supported).  These variables resolve to ports managed by the service proxy.

## The Kubernetes Control Plane

The Kubernetes control plane is split into a set of components, but they all run on a single _master_ node.  These work together to provide a unified view of the cluster.

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

## GCE Cluster Configuration

The scripts and data in the `cluster/` directory automates creating a set of Google Compute Engine VMs and installing all of the Kubernetes components.  There is a single master node and a set of worker (called minion) nodes.

`config-default.sh` has a set of tweakable definitions/parameters for the cluster.

The heavy lifting of configuring the VMs is done by [SaltStack](http://www.saltstack.com/).

The bootstrapping works like this:

1. The `kube-up.sh` script uses the GCE [`startup-script`](https://developers.google.com/compute/docs/howtos/startupscript) mechanism for both the master node and the minion nodes.
  * For the minion, this simply configures and installs SaltStack.  The network range that this minion is assigned is baked into the startup-script for that minion (see [the networking doc](docs/networking.md) for more details).
  * For the master, the release files are staged and then downloaded from GCS and unpacked.  Various parts (specifically the SaltStack configuration) are installed in the right places.  Binaries are included in these tar files.
2. SaltStack then installs the necessary servers on each node.
  * The custom networking bridge is configured on each minion before Docker is installed.
  * Configuration (like telling the `apiserver` the hostnames of the minions) is dynamically created during the saltstack install.
3. After the VMs are started, the `kube-up.sh` script will call `curl` every 2 seconds until the `apiserver` starts responding.

`kube-down.sh` can be used to tear the entire cluster down.  If you build a new release and want to update your cluster, you can use `kube-push.sh` to update and apply (`highstate` in salt parlance) the salt config.

### Cluster Security

As there is no security currently built into the `apiserver`, the salt configuration will install `nginx`.  `nginx` is configured to serve HTTPS with a self signed certificate.  HTTP basic auth is used from the client to `nginx`.  `nginx` then forwards the request on to the `apiserver` over plain old HTTP.  As part of cluster spin up, ssh is used to download both the public cert for the server and a client cert pair.  These are used for mutual authentication to nginx.

All communication within the cluster (worker nodes to the master, for instance) occurs on the internal virtual network and should be safe from eavesdropping.

The password is generated randomly as part of the `kube-up.sh` script and stored in `~/.kubernetes_auth`.
