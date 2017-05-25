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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/maintenance.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Node Maintenance

## Abstract

A proposal for implementing safe node maintenance primitives, in stages.

## Overview

Nodes need periodic maintenance, most of which require partial or total
unavailability to pods.  The cluster admin would like for this to happen (1)
automatically, and (2) respecting any service quality constraints.

Examples of maintenance:

- unmount/remount filesystems to change mkfs params
- remove from service but leave up for debugging
- reboot for kernel or image upgrade
- remove from service for hardware repair
- gracefully reducing the size of the cluster
- take down all machines in a rack for maintenance
- Update Docker (or other container managers that can't restart gracefully)
  without reboot

## Service Quality

This may be thought of as the cluster admin offering a guarantee or SLO, or as
the user specifying their requirements.  Around evictions, there are typically
three parameters:

- Time between evictions of pods in the same replica set.  This lets service
  owners control how frequently they will be serving with relatively cold
  caches.
- Minimum shard strength (fraction of pods within a replica set healthy).  This
  provides an upper bound on administratively imposed reduction of capacity and
  makes it easier to provision.
- Eviction notice.  Service owners can set this to give them enough time to
  write checkpoint files, remove themselves from load balancer or master
  election lists (i.e. to prevent new work from arriving), and finish
  outstanding requests.

## Implementations

In order of increasing complexity (and benefit).

### Drain one node

This is implemented in https://github.com/kubernetes/kubernetes/pull/16698

### Client-driven drain of multiple nodes

In this case, we try to meet some specific service goals specified by the admin
on the command line.  The specification of these parameters only when we run the
tool lets us provide the possibility of safety while avoiding questions of where
to configure and store such specification (treated further down).

```
kubectl drain \
  --min-shard-strength=0.75 \
  --min-seconds-between-evictions=900 \
  --grace=900 $nodes
```

This would still be pretty simple.  We offer no guarantees about safety in the
face of crashes or restarts, but we do meet the needs of admins who want
something more sophisticated than `sleep`.  If kubectl was interrupted and then
restarted with the same arguments, the end state would be exactly the same and
we'd never break the shard strength rule, but the evictions might be more
closely spaced.


### Implementation of the above as a controller

The parameters of a specific cordon, drain, or uncordon request could be pushed
to the API server as a drain object.  The drain controller would have to manage
multiple drains in flight at once.  If the drains have differing parameters,
then more aggressive drains would generally starve safer ones.  While this could
be a useful feature in emergencies, it will be a lot easier to reason about
multiple drains in progress if they all share the same parameters, which would
be configured via kube-env and would represent the cluster admin's SLO.

The drain object would thus look something like this

```
type DrainSpec struct {
  Nodes        NodeList
  Action       string    // one of "Cordon", "Drain", or "Uncordon"
  GracePeriod  int
}
```

A drain object is similar to a deployment:  a deployment expresses how to manage
the transition from one ReplicationController to another, while a drain
expresses how to manage the transition from one set of node state to another.  A
drain controller would watch for drain objects and, within constraints, work
toward the desired end state.  The action of the drain controller can be paused
or stopped by deleting the drain object, and restarted by re-posting it.

A central drain controller would have several advantages: safe restarts and the
ability to react to other events in order to preserve the desired SLO.  For
example, if a machine not in the list of RemoveNodes crashes and takes out other
pods in replication sets also impacted by the drain, it can hold off on
evictions impacting that replication set until MinSecondsBetweenEvictions and
MinShardStrength are OK again.

## Implementation of safety logic

The implementation is the same whether there is one drain in flight or several.
The drain agent (either kubectl or the drain controller) must watch for
replication controllers and pods, and it will watch all pods to notice any time
there is a disruption to a ReplicaSet.  Because there is no reliable mechanism
to replay past events, when the controller restarts, it must wait until it
builds up enough history to establish that it can enforce the safety
constraints.  The standard loop on draining machines will be:

- Is it safe to perform any evictions now?  If so, do them.
- Are any machines empty?  If so mark them done.
- Are any drains done?  If so mark them done.
- Calculate the soonest we can do another eviction.
- Sleep until then, or until we receive a watch update.

## Open Issues

This section describes weaknesses in this proposal not yet addressed.

### Capacity Protection

The process only protects the pods of currently running Replication Controllers.
If a user deletes an rc, by accident or on purpose, and then re-adds it, they
may find that the capacity they were just running in is now gone.  Possible ways
to address this:

- rate limit on pod eviction (also works as defense in depth with the ideas
  below)
- limit max nodes in flight (cluster-wide config)
- alternatively, allow user to specify minimum nodes up
- specify a maximum amount of down capacity, e.g. if the admin specifies 0.2,
  then nothing proceeds if we have, in any resource dimension, less than 80% of
  the cluster's total capacity online in uncordoned, alive nodes

### Maintenance Workflow

The Drain one node implementation easily composes into simple workflows from the
command line.  E.g.

```
% for node in $nodes; do
for> kubectl drain node $node
for> ssh root@$node reboot
for> fping -r 19 $node || break
for> kubectl uncordon node $node
for> done
```

The other implementations do not make it easy to inject workflow.

## Future

### Varying SLOs

We could implement multiple SLOs, either by allowing the cluster admin to
specify choices or by allowing this to be specified in the ReplicaSet or
referring to the similar parameters already present in Deployment.  I believe we
should offer this safety to pods regardless of whether they're managed by
Deployment or not, so I don't think Deployment is the right place to pull this
from.

One proposal is DisruptionBudget (see [issue
12611](http://issues.k8s.io/12611)).  This moves the SLO parameters from
command-line flags (one-shot) or cluster-level config (controller) into API
objects.  Assuming we enforce an upper bound on time between evictions
(otherwise we need potentially unbounded amounts of history), this doesn't
change the implementation.  We'll just have to consult a new source for the SLO
parameters.

### More Sophisticated SLO Expression

Rather than only waiting `X` seconds between evictions, an admin might want to
limit the total in, say, the trailing 6 or 24 hour window.  This requires a
reliable history service, described below.

### Reliable History Service

If we wish our SLO to cover durations longer than an hour or so, the loss of
historical state on restart becomes a real problem.  Adding a reliable history
service to kubernetes would be useful for this, but we could probably also find
other uses.

## References

- https://github.com/kubernetes/kubernetes/issues/3885

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/maintenance.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
