<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Cluster Admin Guide: Cluster Components

This document outlines the various binary components that need to run to
deliver a functioning Kubernetes cluster.

## Master Components

Master components are those that provide the cluster's control plane. For
example, master components are responsible for making global decisions about the
cluster (e.g., scheduling), and detecting and responding to cluster events
(e.g., starting up a new pod when a replication controller's 'replicas' field is
unsatisfied).

Master components could in theory be run on any node in the cluster. However,
for simplicity, current set up scripts typically start all master components on
the same VM, and does not run user containers on this VM. See
[high-availability.md](high-availability.md) for an example multi-master-VM setup.

Even in the future, when Kubernetes is fully self-hosting, it will probably be
wise to only allow master components to schedule on a subset of nodes, to limit
co-running with user-run pods, reducing the possible scope of a
node-compromising security exploit.

### kube-apiserver

[kube-apiserver](kube-apiserver.md) exposes the Kubernetes API; it is the front-end for the
Kubernetes control plane. It is designed to scale horizontally (i.e., one scales
it by running more of them-- [high-availability.md](high-availability.md)).

### etcd

[etcd](etcd.md) is used as Kubernetes' backing store. All cluster data is stored here.
Proper administration of a Kubernetes cluster includes a backup plan for etcd's
data.

### kube-controller-manager

[kube-controller-manager](kube-controller-manager.md) is a binary that runs controllers, which are the
background threads that handle routine tasks in the cluster. Logically, each
controller is a separate process, but to reduce the number of moving pieces in
the system, they are all compiled into a single binary and run in a single
process.

These controllers include:

* Node Controller
 * Responsible for noticing & responding when nodes go down.
* Replication Controller
 * Responsible for maintaining the correct number of pods for every replication
   controller object in the system.
* Endpoints Controller
 * Populates the Endpoints object (i.e., join Services & Pods).
* Service Account & Token Controllers
 * Create default accounts and API access tokens for new namespaces.
* ... and others.

### kube-scheduler

[kube-scheduler](kube-scheduler.md) watches newly created pods that have no node assigned, and
selects a node for them to run on.

### addons

Addons are pods and services that implement cluster features. They don't run on
the master VM, but currently the default setup scripts that make the API calls
to create these pods and services does run on the master VM. See:
[kube-master-addons](http://releases.k8s.io/v1.1.0/cluster/saltbase/salt/kube-master-addons/kube-master-addons.sh)

Addon objects are created in the "kube-system" namespace.

Example addons are:
* [DNS](http://releases.k8s.io/v1.1.0/cluster/addons/dns/) provides cluster local DNS.
* [kube-ui](http://releases.k8s.io/v1.1.0/cluster/addons/kube-ui/) provides a graphical UI for the
  cluster.
* [fluentd-elasticsearch](http://releases.k8s.io/v1.1.0/cluster/addons/fluentd-elasticsearch/) provides
  log storage. Also see the [gcp version](http://releases.k8s.io/v1.1.0/cluster/addons/fluentd-gcp/).
* [cluster-monitoring](http://releases.k8s.io/v1.1.0/cluster/addons/cluster-monitoring/) provides
  monitoring for the cluster.

## Node components

Node components run on every node, maintaining running pods and providing them
the Kubernetes runtime environment.

### kubelet

[kubelet](kubelet.md) is the primary node agent. It:
* Watches for pods that have been assigned to its node (either by apiserver
  or via local configuration file) and:
 * Mounts the pod's required volumes
 * Downloads the pod's secrets
 * Run the pod's containers via docker (or, experimentally, rkt).
 * Periodically executes any requested container liveness probes.
 * Reports the status of the pod back to the rest of the system, by creating a
   "mirror pod" if necessary.
* Reports the status of the node back to the rest of the system.

### kube-proxy

[kube-proxy](kube-proxy.md) enables the Kubernetes service abstraction by maintaining
network rules on the host and performing connection forwarding.

### docker

`docker` is of course used for actually running containers.

### rkt

`rkt` is supported experimentally as an alternative to docker.

### monit

`monit` is a lightweight process babysitting system for keeping kubelet and docker
running.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/cluster-components.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
