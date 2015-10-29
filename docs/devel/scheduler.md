<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# The Kubernetes Scheduler

The Kubernetes scheduler runs as a process alongside the other master
components such as the API server. Its interface to the API server is to watch
for Pods with an empty PodSpec.NodeName, and for each Pod, it posts a Binding
indicating where the Pod should be scheduled.

## The scheduling process

The scheduler tries to find a node for each Pod, one at a time, as it notices
these Pods via watch. There are three steps. First it applies a set of "predicates" that filter out
inappropriate nodes. For example, if the PodSpec specifies resource requests, then the scheduler
will filter out nodes that don't have at least that much resources available (computed
as the capacity of the node minus the sum of the resource requests of the containers that
are already running on the node). Second, it applies a set of "priority functions"
that rank the nodes that weren't filtered out by the predicate check. For example,
it tries to spread Pods across nodes while at the same time favoring the least-loaded
nodes (where "load" here is sum of the resource requests of the containers running on the node,
divided by the node's capacity).
Finally, the node with the highest priority is chosen
(or, if there are multiple such nodes, then one of them is chosen at random). The code
for this main scheduling loop is in the function `Schedule()` in
[plugin/pkg/scheduler/generic_scheduler.go](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/generic_scheduler.go)

## Scheduler extensibility

The scheduler is extensible: the cluster administrator can choose which of the pre-defined
scheduling policies to apply, and can add new ones. The built-in predicates and priorities are
defined in [plugin/pkg/scheduler/algorithm/predicates/predicates.go](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/algorithm/predicates/predicates.go) and
[plugin/pkg/scheduler/algorithm/priorities/priorities.go](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/algorithm/priorities/priorities.go), respectively.
The policies that are applied when scheduling can be chosen in one of two ways. Normally,
the policies used are selected by the functions `defaultPredicates()` and `defaultPriorities()` in
[plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go).
However, the choice of policies
can be overridden by passing the command-line flag `--policy-config-file` to the scheduler, pointing to a JSON
file specifying which scheduling policies to use. See
[examples/scheduler-policy-config.json](../../examples/scheduler-policy-config.json) for an example
config file. (Note that the config file format is versioned; the API is defined in
[plugin/pkg/scheduler/api](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/api/)).
Thus to add a new scheduling policy, you should modify predicates.go or priorities.go,
and either register the policy in `defaultPredicates()` or `defaultPriorities()`, or use a policy config file.

## Exploring the code

If you want to get a global picture of how the scheduler works, you can start in
[plugin/cmd/kube-scheduler/app/server.go](http://releases.k8s.io/release-1.1/plugin/cmd/kube-scheduler/app/server.go)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/scheduler.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
