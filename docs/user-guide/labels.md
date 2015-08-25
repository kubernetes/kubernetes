<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/labels.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Labels

_Labels_ are key/value pairs that are attached to objects, such as pods.
Labels are intended to be used to specify identifying attributes of objects that are meaningful and relevant to users, but which do not directly imply semantics to the core system.
Labels can be used to organize and to select subsets of objects.  Labels can be attached to objects at creation time and subsequently added and modified at any time.
Each object can have a set of key/value labels defined.  Each Key must be unique for a given object.

```json
"labels": {
  "key1" : "value1",
  "key2" : "value2"
}
```

We'll eventually index and reverse-index labels for efficient queries and watches, use them to sort and group in UIs and CLIs, etc. We don't want to pollute labels with non-identifying, especially large and/or structured, data. Non-identifying information should be recorded using [annotations](annotations.md).


## Motivation

Labels enable users to map their own organizational structures onto system objects in a loosely coupled fashion, without requiring clients to store these mappings.

Service deployments and batch processing pipelines are often multi-dimensional entities (e.g., multiple partitions or deployments, multiple release tracks, multiple tiers, multiple micro-services per tier). Management often requires cross-cutting operations, which breaks encapsulation of strictly hierarchical representations, especially rigid hierarchies determined by the infrastructure rather than by users.

Example labels:

   * `"release" : "stable"`, `"release" : "canary"`, ...
   * `"environment" : "dev"`, `"environment" : "qa"`, `"environment" : "production"`
   * `"tier" : "frontend"`, `"tier" : "backend"`, `"tier" : "middleware"`
   * `"partition" : "customerA"`, `"partition" : "customerB"`, ...
   * `"track" : "daily"`, `"track" : "weekly"`

These are just examples; you are free to develop your own conventions.


## Syntax and character set

_Labels_ are key value pairs. Valid label keys have two segments: an optional prefix and name, separated by a slash (`/`).  The name segment is required and must be 63 characters or less, beginning and ending with an alphanumeric character (`[a-z0-9A-Z]`) with dashes (`-`), underscores (`_`), dots (`.`), and alphanumerics between.  The prefix is optional.  If specified, the prefix must be a DNS subdomain: a series of DNS labels separated by dots (`.`), not longer than 253 characters in total, followed by a slash (`/`).
If the prefix is omitted, the label key is presumed to be private to the user. Automated system components (e.g. `kube-scheduler`, `kube-controller-manager`, `kube-apiserver`, `kubectl`, or other third-party automation) which add labels to end-user objects must specify a prefix.  The `kubernetes.io/` prefix is reserved for Kubernetes core components.

Valid label values must be 63 characters or less and must be empty or begin and end with an alphanumeric character (`[a-z0-9A-Z]`) with dashes (`-`), underscores (`_`), dots (`.`), and alphanumerics between.

## Label selectors

Unlike [names and UIDs](identifiers.md), labels do not provide uniqueness. In general, we expect many objects to carry the same label(s).

Via a _label selector_, the client/user can identify a set of objects. The label selector is the core grouping primitive in Kubernetes.

The API currently supports two types of selectors: _equality-based_ and _set-based_.
A label selector can be made of multiple _requirements_ which are comma-separated. In the case of multiple requirements, all must be satisfied so comma separator acts as an AND logical operator.

An empty label selector (that is, one with zero requirements) selects every object in the collection.

### _Equality-based_ requirement

_Equality-_ or _inequality-based_ requirements allow filtering by label keys and values. Matching objects must have all of the specified labels (both keys and values), though they may have additional labels as well.
Three kinds of operators are admitted `=`,`==`,`!=`. The first two represent _equality_ and are simply synonyms. While the latter represents _inequality_. For example:

```
environment = production
tier != frontend
```

The former selects all resources with key equal to `environment` and value equal to `production`.
The latter selects all resources with key equal to `tier` and value distinct from `frontend`.
One could filter for resources in `production` but not `frontend` using the comma operator: `environment=production,tier!=frontend`


### _Set-based_ requirement

_Set-based_ label requirements allow filtering keys according to a set of values. Matching objects must have all of the specified labels (i.e. all keys and at least one of the values specified for each key). Three kind of operators are supported: `in`,`notin` and exists (only the key identifier). For example:

```
environment in (production, qa)
tier notin (frontend, backend)
partition
```

The first example selects all resources with key equal to `environment` and value equal to `production` or `qa`.
The second example selects all resources with key equal to `tier` and value other than `frontend` and `backend`.
The third example selects all resources including a label with key `partition`; no values are checked.
Similarly the comma separator acts as an _AND_ operator for example filtering resource with a `partition` key (not matter the value) and with `environment` different than  `qa`. For example: `partition,environment notin (qa)`.
The _set-based_ label selector is a general form of equality since `environment=production` is equivalent to `environment in (production)`; similarly for `!=` and `notin`.

_Set-based_ requirements can be mixed with _equality-based_ requirements. For example: `partition in (customerA, customerB),environment!=qa`.


## API

LIST and WATCH operations may specify label selectors to filter the sets of objects returned using a query parameter. Both requirements are permitted:

   * _equality-based_ requirements: `?labelSelector=key1%3Dvalue1,key2%3Dvalue2`
   * _set-based_ requirements: `?labelSelector=key+in+%28value1%2Cvalue2%29%2Ckey2+notin+%28value3%29`

Kubernetes also currently supports two objects that use label selectors to keep track of their members, `service`s and `replicationcontroller`s:

* `service`: A [service](services.md) is a configuration unit for the proxies that run on every worker node.  It is named and points to one or more pods.
* `replicationcontroller`: A [replication controller](replication-controller.md) ensures that a specified number of pod "replicas" are running at any one time.

The set of pods that a `service` targets is defined with a label selector. Similarly, the population of pods that a `replicationcontroller` is monitoring is also defined with a label selector. For management convenience and consistency, `services` and `replicationcontrollers` may themselves have labels and would generally carry the labels their corresponding pods have in common.

Sets identified by labels could be overlapping (think Venn diagrams). For instance, a service might target all pods with `"tier": "frontend"` and  `"environment" : "prod"`.  Now say you have 10 replicated pods that make up this tier.  But you want to be able to 'canary' a new version of this component.  You could set up a `replicationcontroller` (with `replicas` set to 9) for the bulk of the replicas with labels `"tier" : "frontend"` and `"environment" : "prod"` and `"track" : "stable"` and another `replicationcontroller` (with `replicas` set to 1) for the canary with labels `"tier" : "frontend"` and  `"environment" : "prod"` and `"track" : "canary"`.  Now the service is covering both the canary and non-canary pods.  But you can mess with the `replicationcontrollers` separately to test things out, monitor the results, etc.

Note that the superset described in the previous example is also heterogeneous. In long-lived, highly available, horizontally scaled, distributed, continuously evolving service applications, heterogeneity is inevitable, due to canaries, incremental rollouts, live reconfiguration, simultaneous updates and auto-scaling, hardware upgrades, and so on.

Pods (and other objects) may belong to multiple sets simultaneously, which enables representation of service substructure and/or superstructure. In particular, labels are intended to facilitate the creation of non-hierarchical, multi-dimensional deployment structures. They are useful for a variety of management purposes (e.g., configuration, deployment) and for application introspection and analysis (e.g., logging, monitoring, alerting, analytics). Without the ability to form sets by intersecting labels, many implicitly related, overlapping flat sets would need to be created, for each subset and/or superset desired, which would lose semantic information and be difficult to keep consistent. Purely hierarchically nested sets wouldn't readily support slicing sets across different dimensions.


## Future developments

Concerning API: we may extend such filtering to DELETE operations in the future.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/labels.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
