# StatefulSets: Running pods which need strong identity and storage

## Motivation

Many examples of clustered software systems require stronger guarantees per instance than are provided
by the Replication Controller (aka Replication Controllers). Instances of these systems typically require:

1. Data per instance which should not be lost even if the pod is deleted, typically on a persistent volume
   * Some cluster instances may have tens of TB of stored data - forcing new instances to replicate data
     from other members over the network is onerous
2. A stable and unique identity associated with that instance of the storage - such as a unique member id
3. A consistent network identity that allows other members to locate the instance even if the pod is deleted
4. A predictable number of instances to ensure that systems can form a quorum
   * This may be necessary during initialization
5. Ability to migrate from node to node with stable network identity (DNS name)
6. The ability to scale up in a controlled fashion, but are very rarely scaled down without human
   intervention

Kubernetes should expose a pod controller (a StatefulSet) that satisfies these requirements in a flexible
manner. It should be easy for users to manage and reason about the behavior of this set. An administrator
with familiarity in a particular cluster system should be able to leverage this controller and its
supporting documentation to run that clustered system on Kubernetes. It is expected that some adaptation
is required to support each new cluster.

This resource is **stateful** because it offers an easy way to link a pod's network identity to its storage
identity and because it is intended to be used to run software that is the holders of state for other
components. That does not mean that all stateful applications *must* use StatefulSets, but the tradeoffs
in this resource are intended to facilitate holding state in the cluster.


## Use Cases

The software listed below forms the primary use-cases for a StatefulSet on the cluster - problems encountered
while adapting these for Kubernetes should be addressed in a final design.

* Quorum with Leader Election
  * MongoDB - in replica set mode forms a quorum with an elected leader, but instances must be preconfigured
    and have stable network identities.
  * ZooKeeper - forms a quorum with an elected leader, but is sensitive to cluster membership changes and
    replacement instances *must* present consistent identities
  * etcd - forms a quorum with an elected leader, can alter cluster membership in a consistent way, and
    requires stable network identities
* Decentralized Quorum
  * Cassandra - allows flexible consistency and distributes data via innate hash ring sharding, is also
    flexible to scaling, more likely to support members that come and go. Scale down may trigger massive
    rebalances.
* Active-active
  * Galera - has multiple active masters which must remain in sync
* Leader-followers
  * Spark in standalone mode - A single unilateral leader and a set of workers


## Background

Replica sets are designed with a weak guarantee - that there should be N replicas of a particular
pod template. Each pod instance varies only by name, and the replication controller errs on the side of
ensuring that N replicas exist as quickly as possible (by creating new pods as soon as old ones begin graceful
deletion, for instance, or by being able to pick arbitrary pods to scale down). In addition, pods by design
have no stable network identity other than their assigned pod IP, which can change over the lifetime of a pod
resource. ReplicaSets are best leveraged for stateless, shared-nothing, zero-coordination,
embarassingly-parallel, or fungible software.

While it is possible to emulate the guarantees described above by leveraging multiple replication controllers
(for distinct pod templates and pod identities) and multiple services (for stable network identity), the
resulting objects are hard to maintain and must be copied manually in order to scale a cluster.

By constrast, a DaemonSet *can* offer some of the guarantees above, by leveraging Nodes as stable, long-lived
entities. An administrator might choose a set of nodes, label them a particular way, and create a
DaemonSet that maps pods to each node. The storage of the node itself (which could be network attached
storage, or a local SAN) is the persistent storage. The network identity of the node is the stable
identity. However, while there are examples of clustered software that benefit from close association to
a node, this creates an undue burden on administrators to design their cluster to satisfy these
constraints, when a goal of Kubernetes is to decouple system administration from application management.


## Design Assumptions

* **Specialized Controller** - Rather than increase the complexity of the ReplicaSet to satisfy two distinct
  use cases, create a new resource that assists users in solving this particular problem.
* **Safety first** - Running a clustered system on Kubernetes should be no harder
  than running a clustered system off Kube. Authors should be given tools to guard against common cluster
  failure modes (split brain, phantom member) to prevent introducing more failure modes. Sophisticated
  distributed systems designers can implement more sophisticated solutions than StatefulSet if necessary -
  new users should not become vulnerable to additional failure modes through an overly flexible design.
* **Controlled scaling** - While flexible scaling is important for some clusters, other examples of clusters
  do not change scale without significant external intervention. Human intervention may be required after
  scaling. Changing scale during cluster operation can lead to split brain in quorum systems. It should be
  possible to scale, but there may be responsibilities on the set author to correctly manage the scale.
* **No generic cluster lifecycle** - Rather than design a general purpose lifecycle for clustered software,
  focus on ensuring the information necessary for the software to function is available. For example,
  rather than providing a "post-creation" hook invoked when the cluster is complete, provide the necessary
  information to the "first" (or last) pod to determine the identity of the remaining cluster members and
  allow it to manage its own initialization.


## Proposed Design

Add a new resource to Kubernetes to represent a set of pods that are individually distinct but each
individual can safely be replaced-- the name **StatefulSet** is chosen to convey that the individual members of
the set are themselves "stateful" and thus each one is preserved. Each member has an identity, and there will
always be a member that thinks it is the "first" one.

The StatefulSet is responsible for creating and maintaining a set of **identities** and ensuring that there is
one pod and zero or more **supporting resources** for each identity. There should never be more than one pod
or unique supporting resource per identity at any one time. A new pod can be created for an identity only
if a previous pod has been fully terminated (reached its graceful termination limit or cleanly exited).

A StatefulSet has 0..N **members**, each with a unique **identity** which is a name that is unique within the
set.

```
type StatefulSet struct {
  ObjectMeta

  Spec StatefulSetSpec
  ...
}

type StatefulSetSpec struct {
  // Replicas is the desired number of replicas of the given template.
  // Each replica is assigned a unique name of the form `name-$replica`
  // where replica is in the range `0 - (replicas-1)`.
  Replicas int

  // A label selector that "owns" objects created under this set
  Selector *LabelSelector

  // Template is the object describing the pod that will be created - each
  // pod created by this set will match the template, but have a unique identity.
  Template *PodTemplateSpec

  // VolumeClaimTemplates is a list of claims that members are allowed to reference.
  // The StatefulSet controller is responsible for mapping network identities to
  // claims in a way that maintains the identity of a member. Every claim in
  // this list must have at least one matching (by name) volumeMount in one
  // container in the template. A claim in this list takes precedence over
  // any volumes in the template, with the same name.
  VolumeClaimTemplates []PersistentVolumeClaim

  // ServiceName is the name of the service that governs this StatefulSet.
  // This service must exist before the StatefulSet, and is responsible for
  // the network identity of the set. Members get DNS/hostnames that follow the
  // pattern: member-specific-string.serviceName.default.svc.cluster.local
  // where "member-specific-string" is managed by the StatefulSet controller.
  ServiceName string
}
```

Like a replication controller, a StatefulSet may be targeted by an autoscaler. The StatefulSet makes no assumptions
about upgrading or altering the pods in the set for now - instead, the user can trigger graceful deletion
and the StatefulSet will replace the terminated member with the newer template once it exits. Future proposals
may offer update capabilities. A StatefulSet requires RestartAlways pods. The addition of forgiveness may be
necessary in the future to increase the safety of the controller recreating pods.


### How identities are managed

A key question is whether scaling down a StatefulSet and then scaling it back up should reuse identities. If not,
scaling down becomes a destructive action (an admin cannot recover by scaling back up). Given the safety
first assumption, identity reuse seems the correct default. This implies that identity assignment should
be deterministic and not subject to controller races (a controller that has crashed during scale up should
assign the same identities on restart, and two concurrent controllers should decide on the same outcome
identities).

The simplest way to manage identities, and easiest to understand for users, is a numeric identity system
starting at I=0 that ranges up to the current replica count and is contiguous.

Future work:

* Cover identity reclamation - cleaning up resources for identities that are no longer in use.
* Allow more sophisticated identity assignment - instead of `{name}-{0 - replicas-1}`, allow subsets and
  complex indexing.

### Controller behavior.

When a StatefulSet is scaled up, the controller must create both pods and supporting resources for
each new identity. The controller must create supporting resources for the pod before creating the
pod. If a supporting resource with the appropriate name already exists, the controller should treat that as
creation succeeding. If a supporting resource cannot be created, the controller should flag an error to
status, back-off (like a scheduler or replication controller), and try again later. Each resource created
by a StatefulSet controller must have a set of labels that match the selector, support orphaning, and have a
controller back reference annotation identifying the owning StatefulSet by name and UID.

When a StatefulSet is scaled down, the pod for the removed indentity should be deleted. It is less clear what the
controller should do to supporting resources. If every pod requires a PV, and a user accidentally scales
up to N=200 and then back down to N=3, leaving 197 PVs lying around may be undesirable (potential for
abuse). On the other hand, a cluster of 5 that is accidentally scaled down to 3 might irreparably destroy
the cluster if the PV for identities 4 and 5 are deleted (may not be recoverable). For the initial proposal,
leaving the supporting resources is the safest path (safety first) with a potential future policy applied
to the StatefulSet for how to manage supporting resources (DeleteImmediately, GarbageCollect, Preserve).

The controller should reflect summary counts of resources on the StatefulSet status to enable clients to easily
understand the current state of the set.

### Parameterizing pod templates and supporting resources

Since each pod needs a unique and distinct identity, and the pod needs to know its own identity, the
StatefulSet must allow a pod template to be parameterized by the identity assigned to the pod. The pods that
are created should be easily identified by their cluster membership.

Because that pod needs access to stable storage, the StatefulSet may specify a template for one or more
**persistent volume claims** that can be used for each distinct pod. The name of the volume claim must
match a volume mount within the pod template.

Future work:

* In the future other resources may be added that must also be templated - for instance, secrets (unique secret per member), config data (unique config per member), and in the futher future, arbitrary extension resources.
* Consider allowing the identity value itself to be passed as an environment variable via the downward API
* Consider allowing per identity values to be specified that are passed to the pod template or volume claim.


### Accessing pods by stable network identity

In order to provide stable network identity, given that pods may not assume pod IP is constant over the
lifetime of a pod, it must be possible to have a resolvable DNS name for the pod that is tied to the
pod identity. There are two broad classes of clustered services - those that require clients to know
all members of the cluster (load balancer intolerant) and those that are amenable to load balancing.
For the former, clients must also be able to easily enumerate the list of DNS names that represent the
member identities and access them inside the cluster. Within a pod, it must be possible for containers
to find and access that DNS name for identifying itself to the cluster.

Since a pod is expected to be controlled by a single controller at a time, it is reasonable for a pod to
have a single identity at a time. Therefore, a service can expose a pod by its identity in a unique
fashion via DNS by leveraging information written to the endpoints by the endpoints controller.

The end result might be DNS resolution as follows:

```
# service mongo pointing to pods created by StatefulSet mdb, with identities mdb-1, mdb-2, mdb-3

dig mongodb.namespace.svc.cluster.local +short A
172.130.16.50

dig mdb-1.mongodb.namespace.svc.cluster.local +short A
# IP of pod created for mdb-1

dig mdb-2.mongodb.namespace.svc.cluster.local +short A
# IP of pod created for mdb-2

dig mdb-3.mongodb.namespace.svc.cluster.local +short A
# IP of pod created for mdb-3
```

This is currently implemented via an annotation on pods, which is surfaced to endpoints, and finally
surfaced as DNS on the service that exposes those pods.

```
// The pods created by this StatefulSet will have the DNS names "mysql-0.NAMESPACE.svc.cluster.local"
// and "mysql-1.NAMESPACE.svc.cluster.local"
kind: StatefulSet
metadata:
  name: mysql
spec:
  replicas: 2
  serviceName: db
  template:
    spec:
      containers:
      - image: mysql:latest

// Example pod created by stateful set
kind: Pod
metadata:
  name: mysql-0
  annotations:
    pod.beta.kubernetes.io/hostname: "mysql-0"
    pod.beta.kubernetes.io/subdomain: db
spec:
  ...
```


### Preventing duplicate identities

The StatefulSet controller is expected to execute like other controllers, as a single writer.  However, when
considering designing for safety first, the possibility of the controller running concurrently cannot
be overlooked, and so it is important to ensure that duplicate pod identities are not achieved.

There are two mechanisms to acheive this at the current time. One is to leverage unique names for pods
that carry the identity of the pod - this prevents duplication because etcd 2 can guarantee single
key transactionality. The other is to use the status field of the StatefulSet to coordinate membership
information. It is possible to leverage both at this time, and encourage users to not assume pod
name is significant, but users are likely to take what they can get. A downside of using unique names
is that it complicates pre-warming of pods and pod migration - on the other hand, those are also
advanced use cases that might be better solved by another, more specialized controller (a
MigratableStatefulSet).


### Managing lifecycle of members

The most difficult aspect of managing a member set is ensuring that all members see a consistent configuration
state of the set. Without a strongly consistent view of cluster state, most clustered software is
vulnerable to split brain. For example, a new set is created with 3 members. If the node containing the
first member is partitioned from the cluster, it may not observe the other two members, and thus create its
own cluster of size 1. The other two members do see the first member, so they form a cluster of size 3.
Both clusters appear to have quorum, which can lead to data loss if not detected.

StatefulSets should provide basic mechanisms that enable a consistent view of cluster state to be possible,
and in the future provide more tools to reduce the amount of work necessary to monitor and update that
state.

The first mechanism is that the StatefulSet controller blocks creation of new pods until all previous pods
are reporting a healthy status. The StatefulSet controller uses the strong serializability of the underyling
etcd storage to ensure that it acts on a consistent view of the cluster membership (the pods and their)
status, and serializes the creation of pods based on the health state of other pods. This simplifies
reasoning about how to initialize a StatefulSet, but is not sufficient to guarantee split brain does not
occur.

The second mechanism is having each "member" use the state of the cluster and transform that into cluster
configuration or decisions about membership. This is currently implemented using a side car container
that watches the master (via DNS today, although in the future this may be to endpoints directly) to
receive an ordered history of events, and then applying those safely to the configuration. Note that
for this to be safe, the history received must be strongly consistent (must be the same order of
events from all observers) and the config change must be bounded (an old config version may not
be allowed to exist forever). For now, this is known as a 'babysitter' (working name) and is intended
to help identify abstractions that can be provided by the StatefulSet controller in the future.


## Future Evolution

Criteria for advancing to beta:

* StatefulSets do not accidentally lose data due to cluster design - the pod safety proposal will
  help ensure StatefulSets can guarantee **at most one** instance of a pod identity is running at
  any time.
* A design consensus is reached on StatefulSet upgrades.

Criteria for advancing to GA:

* StatefulSets solve 80% of clustered software configuraton with minimal input from users and are safe from common split brain problems
  * Several representative examples of StatefulSets from the community have been proven/tested to be "correct" for a variety of partition problems (possibly via Jepsen or similar)
  * Sufficient testing and soak time has been in place (like for Deployments) to ensure the necessary features are in place.
* StatefulSets are considered easy to use for deploying clustered software for common cases

Requested features:

* IPs per member for clustered software like Cassandra that cache resolved DNS addresses that can be used outside the cluster
  * Individual services can potentially be used to solve this in some cases.
* Send more / simpler events to each pod from a central spot via the "signal API"
* Persistent local volumes that can leverage local storage
* Allow pods within the StatefulSet to identify "leader" in a way that can direct requests from a service to a particular member.
* Provide upgrades of a StatefulSet in a controllable way (like Deployments).


## Overlap with other proposals

* Jobs can be used to perform a run-once initialization of the cluster
* Init containers can be used to prime PVs and config with the identity of the pod.
* Templates and how fields are overriden in the resulting object should have broad alignment
* DaemonSet defines the core model for how new controllers sit alongside replication controller and
  how upgrades can be implemented outside of Deployment objects.


## History

StatefulSets were formerly known as PetSets and were renamed to be less "cutesy" and more descriptive as a
prerequisite to moving to beta. No animals were harmed in the making of this proposal.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/stateful-apps.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
