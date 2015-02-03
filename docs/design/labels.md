# Labels

_Labels_ are key/value pairs identifying client/user-defined attributes (and non-primitive system-generated attributes) of API objects, which are stored and returned as part of the [metadata of those objects](/docs/api-conventions.md). Labels can be used to organize and to select subsets of objects according to these attributes.

Each object can have a set of key/value labels set on it, with at most one label with a particular key. 
```
"labels": {
  "key1" : "value1",
  "key2" : "value2"
}
```

Unlike [names and UIDs](/docs/identifiers.md), labels do not provide uniqueness. In general, we expect many objects to carry the same label(s). 

Via a _label selector_, the client/user can identify a set of objects. The label selector is the core grouping primitive in Kubernetes. 

Label selectors may also be used to associate policies with sets of objects.

We also [plan](https://github.com/GoogleCloudPlatform/kubernetes/issues/560) to make labels available inside pods and [lifecycle hooks](/docs/container-environment.md).

Valid label keys are comprised of two segments - prefix and name - separated
by a slash (`/`).  The name segment is required and must be a DNS label: 63
characters or less, all lowercase, beginning and ending with an alphanumeric
character (`[a-z0-9]`), with dashes (`-`) and alphanumerics between.  The
prefix and slash are optional.  If specified, the prefix must be a DNS
subdomain (a series of DNS labels separated by dots (`.`), not longer than 253
characters in total.

If the prefix is omitted, the label key is presumed to be private to the user.
System components which use labels must specify a prefix.  The `kubernetes.io`
prefix is reserved for kubernetes core components.

## Motivation

Service deployments and batch processing pipelines are often multi-dimensional entities (e.g., multiple partitions or deployments, multiple release tracks, multiple tiers, multiple micro-services per tier). Management often requires cross-cutting operations, which breaks encapsulation of strictly hierarchical representations, especially rigid hierarchies determined by the infrastructure rather than by users. Labels enable users to map their own organizational structures onto system objects in a loosely coupled fashion, without requiring clients to store these mappings.

## Label selectors

Label selectors permit very simple filtering by label keys and values. The simplicity of label selectors is deliberate. It is intended to facilitate transparency for humans, easy set overlap detection, efficient indexing, and reverse-indexing (i.e., finding all label selectors matching an object's labels - https://github.com/GoogleCloudPlatform/kubernetes/issues/1348). 

Currently the system supports selection by exact match of a map of keys and values. Matching objects must have all of the specified labels (both keys and values), though they may have additional labels as well.

We are in the process of extending the label selection specification (see [selector.go](/pkg/labels/selector.go) and https://github.com/GoogleCloudPlatform/kubernetes/issues/341) to support conjunctions of requirements of the following forms:
```
key1 in (value11, value12, ...)
key1 not in (value11, value12, ...)
key1 exists
```

LIST and WATCH operations may specify label selectors to filter the sets of objects returned using a query parameter: `?labels=key1%3Dvalue1,key2%3Dvalue2,...`. We may extend such filtering to DELETE operations in the future.

Kubernetes also currently supports two objects that use label selectors to keep track of their members, `service`s and `replicationController`s:
- `service`: A [service](/docs/services.md) is a configuration unit for the proxies that run on every worker node.  It is named and points to one or more pods.
- `replicationController`: A [replication controller](/docs/replication-controller.md) ensures that a specified number of pod "replicas" are running at any one time.  If there are too many, it'll kill some.  If there are too few, it'll start more.

The set of pods that a `service` targets is defined with a label selector. Similarly, the population of pods that a `replicationController` is monitoring is also defined with a label selector. 

For management convenience and consistency, `services` and `replicationControllers` may themselves have labels and would generally carry the labels their corresponding pods have in common.

In the future, label selectors will be used to identify other types of distributed service workers, such as worker pool members or peers in a distributed application.

Individual labels are used to specify identifying metadata, and to convey the semantic purposes/roles of pods of containers. Examples of typical pod label keys include `service`, `environment` (e.g., with values `dev`, `qa`, or `production`), `tier` (e.g., with values `frontend` or `backend`), and `track` (e.g., with values `daily` or `weekly`), but you are free to develop your own conventions.

Sets identified by labels and label selectors could be overlapping (think Venn diagrams). For instance, a service might target all pods with `tier in (frontend), environment in (prod)`.  Now say you have 10 replicated pods that make up this tier.  But you want to be able to 'canary' a new version of this component.  You could set up a `replicationController` (with `replicas` set to 9) for the bulk of the replicas with labels `tier=frontend, environment=prod, track=stable` and another `replicationController` (with `replicas` set to 1) for the canary with labels `tier=frontend, environment=prod, track=canary`.  Now the service is covering both the canary and non-canary pods.  But you can mess with the `replicationControllers` separately to test things out, monitor the results, etc. 

Note that the superset described in the previous example is also heterogeneous. In long-lived, highly available, horizontally scaled, distributed, continuously evolving service applications, heterogeneity is inevitable, due to canaries, incremental rollouts, live reconfiguration, simultaneous updates and auto-scaling, hardware upgrades, and so on.

Pods (and other objects) may belong to multiple sets simultaneously, which enables representation of service substructure and/or superstructure. In particular, labels are intended to facilitate the creation of non-hierarchical, multi-dimensional deployment structures. They are useful for a variety of management purposes (e.g., configuration, deployment) and for application introspection and analysis (e.g., logging, monitoring, alerting, analytics). Without the ability to form sets by intersecting labels, many implicitly related, overlapping flat sets would need to be created, for each subset and/or superset desired, which would lose semantic information and be difficult to keep consistent. Purely hierarchically nested sets wouldn't readily support slicing sets across different dimensions.

Pods may be removed from these sets by changing their labels. This flexibility may be used to remove pods from service for debugging, data recovery, etc.

Since labels can be set at pod creation time, no separate set add/remove operations are necessary, which makes them easier to use than manual set management. Additionally, since labels are directly attached to pods and label selectors are fairly simple, it's easy for users and for clients and tools to determine what sets they belong to (i.e., they are reversible). OTOH, with sets formed by just explicitly enumerating members, one would (conceptually) need to search all sets to determine which ones a pod belonged to.

## Labels vs. annotations

We'll eventually index and reverse-index labels for efficient queries and watches, use them to sort and group in UIs and CLIs, etc. We don't want to pollute labels with non-identifying, especially large and/or structured, data. Non-identifying information should be recorded using [annotations](/docs/annotations.md).
