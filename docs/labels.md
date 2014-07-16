# Labels

Service deployments and batch processing pipelines are often multi-dimensional entities (e.g., multiple release tracks, multiple tiers, multiple micro-services per tier). Management often requires cross-cutting operations, which breaks encapsulation of strictly hierarchical representations, especially rigid hierarchies determined by the infrastructure rather than by users. 

Therefore, loosely coupled cooperating pods are organized using key/value _labels_.

Individual labels are used to specify identifying metadata, and to convey the semantic purposes/roles of pods of containers. Examples of typical pod label keys include `service`, `environment` (e.g., with values `dev`, `qa`, or `production`), `tier` (e.g., with values `frontend` or `backend`), and `track` (e.g., with values `daily` or `weekly`), but you are free to develop your own conventions.

Each pod can have a set of key/value labels set on it, with at most one label with a particular key. 

Via a "label selector" the user can identify a set of `pods`. The label selector is the core grouping primitive in Kubernetes. It could be used to identify service replicas or shards, worker pool members, or peers in a distributed application.

Kubernetes currently supports two objects that use label selectors to keep track of their members, `service`s and `replicationController`s:
- `service`: A service is a configuration unit for the proxies that run on every worker node.  It is named and points to one or more pods.
- `replicationController`: A replication controller takes a template and ensures that there is a specified number of "replicas" of that template running at any one time.  If there are too many, it'll kill some.  If there are too few, it'll start more.

The set of pods that a `service` targets is defined with a label selector. Similarly, the population of pods that a `replicationController` is monitoring is also defined with a label selector. 

Pods may be removed from these sets by changing their labels. This flexibility may be used to remove pods from service for debugging, data recovery, etc.

For management convenience and consistency, `services` and `replicationControllers` may themselves have labels and would generally carry the labels their corresponding pods have in common.

Sets identified by labels and label selectors could be overlapping (think Venn diagrams). For instance, a service might point to all pods with `tier in (frontend), environment in (prod)`.  Now say you have 10 replicated pods that make up this tier.  But you want to be able to 'canary' a new version of this component.  You could set up a `replicationController` (with `replicas` set to 9) for the bulk of the replicas with labels `tier=frontend, environment=prod, track=stable` and another `replicationController` (with `replicas` set to 1) for the canary with labels `tier=frontend, environment=prod, track=canary`.  Now the service is covering both the canary and non-canary pods.  But you can mess with the `replicationControllers` separately to test things out, monitor the results, etc. 

Note that the superset described in the previous example is also heterogeneous. In long-lived, highly available, horizontally scaled, distributed, continuously evolving service applications, heterogeneity is inevitable, due to canaries, incremental rollouts, live reconfiguration, simultaneous updates and auto-scaling, hardware upgrades, and so on.

Pods may belong to multiple sets simultaneously, which enables representation of service substructure and/or superstructure. In particular, labels are intended to facilitate the creation of non-hierarchical, multi-dimensional deployment structures. They are useful for a variety of management purposes (e.g., configuration, deployment) and for application introspection and analysis (e.g., logging, monitoring, alerting, analytics). Without the ability to form sets by intersecting labels, many implicitly related, overlapping flat sets would need to be created, for each subset and/or superset desired, which would lose semantic information and be difficult to keep consistent. Purely hierarchically nested sets wouldn't readily support slicing sets across different dimensions.

Since labels can be set at pod creation time, no separate set add/remove operations are necessary, which makes them easier to use than manual set management. Additionally, since labels are directly attached to pods and label selectors are fairly simple, it's easy for users and for clients and tools to determine what sets they belong to. OTOH, with sets formed by just explicitly enumerating members, one would (conceptually) need to search all sets to determine which ones a pod belonged to.
