# Kubernetes Design Overview

Kubernetes is a system for managing containerized applications across multiple
hosts, providing basic mechanisms for deployment, maintenance, and scaling of
applications.

Kubernetes establishes robust declarative primitives for maintaining the desired
state requested by the user. We see these primitives as the main value added by
Kubernetes. Self-healing mechanisms, such as auto-restarting, re-scheduling, and
replicating containers require active controllers, not just imperative
orchestration.

Kubernetes is primarily targeted at applications composed of multiple
containers, such as elastic, distributed micro-services. It is also designed to
facilitate migration of non-containerized application stacks to Kubernetes. It
therefore includes abstractions for grouping containers in both loosely coupled
and tightly coupled formations, and provides ways for containers to find and
communicate with each other in relatively familiar ways.

Kubernetes enables users to ask a cluster to run a set of containers. The system
automatically chooses hosts to run those containers on. While Kubernetes's
scheduler is currently very simple, we expect it to grow in sophistication over
time. Scheduling is a policy-rich, topology-aware, workload-specific function
that significantly impacts availability, performance, and capacity. The
scheduler needs to take into account individual and collective resource
requirements, quality of service requirements, hardware/software/policy
constraints, affinity and anti-affinity specifications, data locality,
inter-workload interference, deadlines, and so on. Workload-specific
requirements will be exposed through the API as necessary.

Kubernetes is intended to run on a number of cloud providers, as well as on
physical hosts.

A single Kubernetes cluster is not intended to span multiple availability zones.
Instead, we recommend building a higher-level layer to replicate complete
deployments of highly available applications across multiple zones (see
[the multi-cluster doc](../admin/multi-cluster.md) and [cluster federation proposal](../proposals/federation.md)
for more details).

Finally, Kubernetes aspires to be an extensible, pluggable, building-block OSS
platform and toolkit. Therefore, architecturally, we want Kubernetes to be built
as a collection of pluggable components and layers, with the ability to use
alternative schedulers, controllers, storage systems, and distribution
mechanisms, and we're evolving its current code in that direction. Furthermore,
we want others to be able to extend Kubernetes functionality, such as with
higher-level PaaS functionality or multi-cluster layers, without modification of
core Kubernetes source. Therefore, its API isn't just (or even necessarily
mainly) targeted at end users, but at tool and extension developers. Its APIs
are intended to serve as the foundation for an open ecosystem of tools,
automation systems, and higher-level API layers. Consequently, there are no
"internal" inter-component APIs. All APIs are visible and available, including
the APIs used by the scheduler, the node controller, the replication-controller
manager, Kubelet's API, etc. There's no glass to break -- in order to handle
more complex use cases, one can just access the lower-level APIs in a fully
transparent, composable manner.

For more about the Kubernetes architecture, see [architecture](architecture.md).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
