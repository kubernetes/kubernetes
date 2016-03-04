<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Pod Templates

Pod templates are [pod](pods.md) specifications which are included in other objects, such as
[Replication Controllers](replication-controller.md), [Jobs](jobs.md), and
[DaemonSets](../admin/daemons.md).  Controllers use Pod Templates to make actual pods.

Rather than specifying the current desired state of all replicas, pod templates are like cookie cutters. Once a cookie has been cut, the cookie has no relationship to the cutter. There is no quantum entanglement. Subsequent changes to the template or even switching to a new template has no direct effect on the pods already created. Similarly, pods created by a replication controller may subsequently be updated directly. This is in deliberate contrast to pods, which do specify the current desired state of all containers belonging to the pod. This approach radically simplifies system semantics and increases the flexibility of the primitive.


## Future Work

A replication controller creates new pods from a template, which is currently inline in the `ReplicationController` object, but which we plan to extract into its own resource [#170](http://issue.k8s.io/170).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/pod-templates.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
