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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/bandwidth-resources.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Bandwidth resources

When specifying a [pod](pods.md), you can optionally specify how much bandwidth (bits/sec) each
container can consume.  There are separate constraints for ingress and egress bandwidth. When pods
have resource limits, the scheduler is able to make better
decisions about which nodes to place pods on, and contention for resources can be handled in a
consistent manner.

*ingress* and *egress* bandwidth are each a *resource type*.  All resource types have a base unit.
For both ingress and egress this unit is bits per second.

Ingress and egress are collectively referred to as *bandwidth resources*.  Bandwidth
resources are measureable quantities which can be requested, allocated, and consumed.  They are
distinct from [API resources](working-with-resources.md).  API resources, such as pods and
[services](services.md) are objects that can be written to and retrieved from the Kubernetes API
server.

## Pod resource limits

Each Pod can optionally specify `spec.containers[].resources.limits.ingress` and/or
`spec.containers[].resources.limits.egress`.

Specifying resource limits is optional.  In some clusters, an unset value may be replaced with a
default value when a pod is created or updated.  The default value depends on how the cluster is
configured.

Although limits can only be specified on individual containers, it is convenient to talk about pod
resource limits.  A *pod resource limit* for a particular resource type is the sum of the resource
limits of that type for each container in the pod, with unset values treated as zero.

The following pod has two containers.  Each has a limit of 1Mbit of ingress bandwidth and 2Mbit of
egress bandwidth.
The pod can be said to have a limit of 2Mbit of ingress bandwidth and 4Mbit of egress bandwidth.

*Note* Because of limitations in container runtimes, we can't typically limit the bandwidth between
containers, so different containers in a Pod may steal from each other.  We can (and do) limit
the overall bandwidth for the Pod.  We hope to refine this as capabilities for hierarchical
network namespaces come into the container runtimes like Docker.


```yaml
apiVersion: v1
kind: Pod
metadata:
  name: frontend
spec:
  containers:
  - name: db
    image: mysql
    resources:
      limits:
        ingress: "1M"
        egress: "2M"
  - name: wp
    image: wordpress
    resources:
      limits:
        ingress: "1M"
        egress: "2M"
```

## How pods with resource limits are scheduled

When a pod is created, the Kubernetes scheduler selects a node for the pod to
run on.  Each node has a maximum capacity for each of the resource types: the
amount of CPU and memory it can provide for pods.  The scheduler ensures that,
for each bandwidth resource type (ingress/egress), the sum of the resource requests of the
containers scheduled to the node is less than the capacity of the node.  Note
that even if the actual bandwidth resource usage on nodes is very low, the
scheduler will still refuse to place pods onto nodes if the capacity check
fails.  This protects against a resource shortage on a node when resource usage
later increases, such as due to a daily peak in request rate.

Note: Although the scheduler normally spreads pods out across nodes, there are currently some cases
where pods with no limits (unset values) might all land on the same node.

## How pods with resource limits are run

When kubelet starts a container of a pod, it manages the network resource consumption based
on the pod's IP address, using the `tc` command.

- The `spec.container[].resources.limits.ingress` or `egress` is summed for all containers in the pod.
- The `tc` command is used to create a bandwidth limit classes for ingress and egress. Different classes are used for each.
- The `tc` command is used to apply this bandwidth limit class to all network traffic either originating (egress) or arriving (ingress) to the Pod's IP address.

`tc` is configured to use the hierarchical token bucket (`htb`) algorithm for bandwidth shaping.  Packets are slowed down in the kernel's queues in
order to achieve the desired bandwidth restrictions.

## Monitoring bandwidth resource usage

It is not currently possible to monitor the bandwidth usage of a pod.  Support for this is planned.

## Troubleshooting

### My pods are pending with event message failedScheduling

If the scheduler cannot find any node where a pod can fit, then the pod will remain unscheduled
until a place can be found.    An event will be produced each time the scheduler fails to find a
place for the pod, like this:

```console
$ kubectl describe pods/frontend | grep -A 3 Events
Events:
  FirstSeen				LastSeen			Count	From SubobjectPath	Reason			Message
  Tue, 30 Jun 2015 09:01:41 -0700	Tue, 30 Jun 2015 09:39:27 -0700	128	{scheduler }            failedScheduling	Error scheduling: For each of these fitness predicates, pod frontend failed on at least one node: PodFitsResources.
```

If a pod or pods are pending with this message, then there are several things to try:
- Add more nodes to the cluster.
- Terminate unneeded pods to make room for pending pods.
- Check that the pod is not larger than the nodes.  For example, if all the nodes
have a capacity of `ingress: 10Mi`, then a pod with a limit of `ingress: 11Mi` will never be scheduled.

You can check node capacities with the `kubectl get nodes -o <format>` command.
Here are some example command lines that extract just the necessary information:
- `kubectl get nodes -o yaml | grep '\sname\|cpu\|memory'`
- `kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, cap: .status.capacity}'`

The [resource quota](../admin/resource-quota.md) feature can be configured
to limit the total amount of resources that can be consumed.  If used in conjunction
with namespaces, it can prevent one team from hogging all the resources.

## Planned improvements

Currently we can't tell the difference between traffic from/to different containers in the pod.  Eventually
we would like to have a hierarchical network namespace, and this will allow us to apply different network classes
to each container in the pod, this in turn will enable us to restrict bandwidth to particular containers, rather
than the entire pod.

Additionally, we can use the kernel's network queues to monitor network resources consumed and add that information
to the pod, we would like to add monitoring for this activity as well.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/bandwidth-resources.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
