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
[here](http://releases.k8s.io/release-1.1/docs/proposals/multiple-schedulers.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# PetSets: Running pods which need strong identity and storage

## Open Issues

* Add examples
* Discuss failure modes for various types of clusters
* Provide an active-active example
* Templating proposals need to be argued through to reduce options

## Motivation

Many examples of clustered software systems require stronger guarantees per instance than are provided
by the Replication Controller (aka ReplicaSet). Instances of these systems typically require:

1. Data per instance which should not be lost even if the pod is deleted, typically on a persistent volume
   * Some cluster instances may have tens of TB of stored data - forcing new instances to replicate data
     from other members over the network is onerous
2. A stable and unique identity associated with that instance of the storage - such as a unique member id
3. A consistent network identity that allows other members to locate the instance even if the pod is deleted
4. A predictable number of instances to ensure that systems can form a quorum
   * This may be necessary during initialization
5. Instances can migrate from machine to machine as necessary and are not tied to an instance
6. The ability to scale up in a controlled fashion, but are very rarely scaled down without human
   intervention

Kubernetes should expose a pod controller (a PetSet) that satisfies these requirements in a flexible
manner. It should be easy for users to manage and reason about the behavior of this set. An administrator
with familiarity in a particular cluster system should be able to leverage this controller and its
supporting documentation to run that clustered system on Kubernetes. It is expected that some adaptation
is required to support each new cluster.


## Use Cases

The software listed below forms the primary use-cases for a PetSet on the cluster - problems encountered
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
  * **Need active-active master example - galera? **
* ??? 


## Background

ReplicaSets are designed with a weak guarantee - that there should be N replicas of a particular pod template.
Each pod instance varies only by name, and the replication controller errs on the side of ensuring that N
replicas exist as quickly as possible (by creating new pods as soon as old ones begin graceful deletion, for
instance). In addition, pods by design have no stable network identity other than their assigned pod IP,
which can change over the lifetime of a pod resource. ReplicaSets are best leveraged for shared-nothing,
zero-coordination software.

While it is possible to emulate the guarantees described above by leveraging multiple replication controllers
(for distinct pod templates and pod identities) and multiple services (for stable network identity), the
resulting objects are hard to maintain and must be copied manually in order to scale a cluster.

By constrast, a DaemonSet *can* offer some of the guarantees above, by leveraging Nodes as stable, long
lived entities. An administrator might choose a set of nodes, label them a particular way, and create a
DaemonSet that maps pods to each node. The storage of the node itself (which could be network attached
storage, or a local SAN) is the persistent storage. The network identity of the node is the stable
identity. However, while there are examples of clustered software that benefit from close association to
a node, this creates an undue burden on administrators to design their cluster to satisfy these
constraints, when a goal of Kubernetes is to decouple system administration from application management.


## Design Assumptions

* **Specialized Controller** - Rather than increase the complexity of the ReplicaSet to satisfy two distinct
  use cases, create a new resource that assists users in solving this particular problem.
* **No built-in update** - Updating clustered software can be complex, since existing software may dictate
  certain orchestration occur as each instance is created. For now, assume that updates to the PetSet are
  driven by external or innate orchestration
* **Safety first** - Running a clustered system on Kubernetes should be no harder
  than running a clustered system off Kube. Authors should be given tools to guard against common cluster
  failure modes (split brain, phantom member) to prevent introducing more failure modes. Sophisticated
  distributed systems designers can implement more sophisticated solutions than PetSet if necessary -
  new users should not become vulnerable to additional failure modes through an overly flexible design.
* **Limited scaling** - While flexible scaling is important for some clusters, other examples of clusters
  do not change scale without significant external intervention. Human intervention may be required after
  scaling. Changing scale during cluster operation can lead to split brain in quorum systems. It should be
  possible to scale easily, but the details of making that safe belong to the pods and image authors.
* **No generic cluster lifecycle** - Rather than design a general purpose lifecycle for clustered software,
  focus on ensuring the information necessary for the software to function is available. For example,
  rather than providing a "post-creation" hook invoked when the cluster is complete, provide the necessary
  information to the "first" (or last) pod to determine the identity of the remaining cluster members and
  allow it to manage its own initialization.
* **External access direct to cluster members is out of scope** - exposing pods to consumers that cannot
  access the pod network is out of scope, because we do not currently support headless services being
  exposed via NodePort. A workaround is to allow external clients to access the pod network, or to
  create one NodePort service per member. A future design should cover headless external service access.


## Proposed Design

Add a new resource to Kubernetes to represent a set of pods that are individually distinct but each
individual can safely be replaced-- the name **PetSet** (working name) is chosen to convey that the
individual members of the set are themselves "pets" and thus each one is important. A relevant analogy
is that a PetSet is composed of pets, but the pets are like goldfish. If you have a blue, red, and
yellow goldfish, and the red goldfish dies, you replace it with another red goldfish and no one would
notice. If you suddenly have three red goldfish, someone will notice.

The PetSet is responsible for creating and maintaining a set of **identities** and ensuring that there is
one pod and zero or more **supporting resources** for each identity. There should never be more than one pod
or unique supporting resource per identity at any one time. A new pod can be created for an identity only
if a previous pod has been fully terminated (reached its graceful termination limit or cleanly exited).

```
type PetSet struct {
    ObjectMeta

    Spec PetSetSpec
    ...
}

type PetSetSpec struct {
    // Number of instances in the set
    Replicas *int
}
```

Like a replication controller, a PetSet may be targeted by an autoscaler. The PetSet makes no assumptions
about upgrading or altering the pods in the set (similar to a DaemonSet) - instead, the user can trigger
graceful deletion and the PetSet will replace the terminated member with the newer template once it exits.
Future proposals may offer update capabilities. A PetSet requires RunAlways pods.

### How identities are managed

A key question is whether scaling down a PetSet and then scaling it back up should reuse identities. If not,
scaling down becomes a destructive action (an admin cannot recover by scaling back up). Given the safety
first assumption, identity reuse seems the correct default. This implies that identity assignment should
be deterministic and not subject to controller races (a controller that has crashed during scale up should
assign the same identities on restart, and two concurrent controllers should decide on the same outcome
identities).

The simplest way to manage identities, and easiest to understand for users, is a numeric identity system
starting at I=1 that ranges up to the current replica count and is contiguous.

### Controller behavior.

When a PetSet is scaled up, the controller must create both pods and supporting resources for
each new identity. The controller must create supporting resources for the pod before creating the
pod. If a supporting resource with the appropriate name already exists, the controller should treat that as
creation succeeding. If a supporting resource cannot be created, the controller should flag an error to
status, back-off (like a scheduler or replication controller), and try again later.

When a PetSet is scaled down, the pod for the removed indentity should be deleted. It is less clear what the
controller should do to supporting resources. If every pod requires a PV, and a user accidentally scales
up to N=200 and then back down to N=3, leaving 197 PVs lying around may be undesirable (potential for
abuse). On the other hand, a cluster of 5 that is accidentally scaled down to 3 might irreparably destroy
the cluster if the PV for identities 4 and 5 are deleted (may not be recoverable). For the initial proposal,
leaving the supporting resources is the safest path (safety first) with a potential future policy applied
to the PetSet for how to manage supporting resources (DeleteImmediately, GarbageCollect, Preserve).

OPEN: Should information about the state of the pods in the PetSet be reflected in PetSet status? Summarizing
  which pods match the current PetSet state would be extremely valuable.


### Parameterizing pod templates and supporting resources

Since each pod needs a unique and distinct identity, and the pod needs to know its own identity, the
PetSet must allow a pod template to be parameterized by the identity assigned to the pod. The pods that
are created should be easily identified by their cluster membership.


```
type PetSetSpec struct {
    // Number of instances in the set
    Replicas *int

    // The template to use for the pod in the set
    Template *PodTemplateSpec
}

// Proposal 1a: Use the name of the pod as the identity, use downward API for getting identity
// Downside: all fields that want to leverage the identity need equivalent valueFrom semantics
{
    "spec": {
        "template": {
            "metadata": {"generateName": "pod-name"}, // assume identity is automatically added to pod name
            "spec": {
                "containers": [{
                    "env": [{
                        "name": "IDENTITY",
                        "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}
                        }
                    }]
                }]
            }
        }
    }
}

// Proposal 1b: Add an annotation to each pod when the controller creates it
// Downside: all fields that want to leverage the identity need equivalent valueFrom semantics
{
    "spec": {
        "template": {
            "spec": {
                "containers": [{
                    "env": [{
                        "name": "IDENTITY",
                        "valueFrom": {"fieldRef": {"fieldPath": "metadata.annotations[alpha.kubernetes.io/identity]"}}
                    }]
                }]
            }
        }
    }
}

// Proposal 1c: Add a new field to pod spec that is write-once/read-many - identity
// Downside: all fields that want to leverage the identity need equivalent valueFrom semantics
{
    "spec": {
        "template": {
            "spec": {
                "identity": "", // set on creation
                "containers": [{
                    "env": [{
                        "name": "IDENTITY",
                        "valueFrom": {"fieldRef": {"fieldPath": "spec.identity"}}
                    }]
                }]
            }
        }
    }
}

// Proposal 1d: support template transformation of a field path
// Downside: possibly too flexible, can conflict with existing env resolution semantics, bypasses downward API
{
    "spec": {
        "template": {
            "metadata": {"name": "pod-name-$(IDENTITY)"},
            "spec": {
                "containers": [{
                    "env": [{
                        "name": "IDENTITY",
                        "value": "$(IDENTITY)"
                    }]
                }]
            }
        }
    }
}

// Proposal 1e: specify merge-patch semantics with some variable replacement
// Downside: merge patch is inflexible to structural template edits
{
    "spec": {
        "template": {
            "metadata": {},
            "spec": {
                "containers": [{
                    "env": [{
                        "name": "IDENTITY",
                    }]
                }]
            }
        },
        "templateTransform": [
            {"op": "add", "path": "/metadata/name", "value": "pod-name-$(IDENTITY)"},
            {"op": "add", "path": "/spec/containers[0]/env[0]/value", "value": "$(IDENTITY)"}
        ]
    }
}
```

Proposal 1e has the most overlap with other concepts already existing in the API, and makes it easy
for clients to identify what will be transformed per template. Outside-in transformations also are
most amenable to automatic generation.

Because that pod needs access to stable storage, the PetSet may specify a template for one or more
**supporting resources** that can be used for each distinct pod. In the future other resources may be
added that must also be templated - for instance, secrets (unique secret per member), config data (unique
config per member), and in the future, arbitrary extension resources.

The set of parameterization allowed to each pod and supporting resources is intentionally limited to those
that accomplish the goals above. It should be possible to:

* Identify the supporting resources by name in the pod template (pvc name)
* Give the supporting resource a predictable name based on identity
* Set environment variables that correspond to the identity of the pod for use by the software
* Provide the current desired set size (`spec.replicas`) to each pod as an integer

```
type PetSetSpec struct {
    ...
    // The template to use for the pod in the set
    Template *PodTemplateSpec
    SupportTemplates []SupportTemplateSpec
}

type SupportTemplateSpec struct {
    // only one of these fields may be set, like Volumes
    PersistentVolumeClaim *PersistentVolumeClaimTemplateSpec
    Secret *SecretTemplateSpec
    ConfigData *ConfigDataTemplateSpec
    // anything not supported by the above
    Object runtime.Object
}

// Proposal 2a: use the name of the support resource as identity implicitly
// Downside: Requires the controller to understand the places to be parameterized in the template - prevents
//   custom resources from being supported.
{
    "spec": {
        "template": {
            "spec": {
                "volumes": [{
                    "persistentVolumeClaim": {
                        "claimName": "pod-name-pvc" // controller appends IDENTITY to this claim automatically
                    }
                }]
            }
        },
        "supportTemplates": [{
            "persistentVolumeClaim": {
                "metadata": {"name": "pod-name-pvc"} // controller appends IDENTTY to this claim automatically
            }
        }]
    }
}

// Proposal 2b: append identity to specified field paths
// Downside: Flexibility is limited, may need "operations" on the field in order to transform value
{
    "spec": {
        // ...

        "supportTemplates": [{
            "persistentVolumeClaim": {
                "metadata": {"name": "pod-name-pvc-"}
            },
            "fieldPaths": ["metadata.name"] // controller appends IDENTTY to field metadata.name
        }]
    }
}

// Proposal 2c: append identity to specified field paths, name is special
// Downside: Flexibility is limited, transformations may be required
{
    "spec": {
        // ...

        "supportTemplates": [{
            "configData": {
                "metadata": {"generateName": "pod-name-pvc-"}
                // controller appends IDENTITY to field metadata.generateName and sets metadata.name from that
            },
            "fieldPaths": ["data[identity]"] // controller appends IDENTITY to the "identity" key of config data
        }]
    }
}

// Proposal 2d: support template transformation of a field path
// Downside: Unrestricted use possibly too powerful
// Downside: Harder for clients to recognize what fields are parameterized
{
    "spec": {
        "template": {
            "spec": {
                "volumes": [{
                    "persistentVolumeClaim": {
                        "claimName": "pod-name-pvc" // controller appends IDENTITY to this claim automatically
                    }
                }]
            }
        },
        "supportTemplates": [{
            "persistentVolumeClaim": {
                "metadata": {"name": "pod-name-pvc-$(IDENTITY)"} // controller adds IDENTTY to this claim as defined
            }
        }]
    }
}

// Proposal 2e: specify merge-patch semantics with some variable replacement
{
    "spec": {
        // ...

        "supportTemplates": [{
            "persistentVolumeClaim": {
                "spec": {...}
            },
            "transform": [{"op": "add", "path": "/metadata/name", "value": "pod-name-pvc-$(IDENTITY)"}]
        }]
    }
}

```

Of all proposals, 2e is the closest to actual semantics we support today (variable replacement in EnvVar and
patch operations). The path syntax from RFC6902 is not optimal (we prefer JQ field path semantics), but we
could consider supporting both. Since we desire to support an integer value for quorum size, we may need to
transform the set size value to an integer for replacement into a supporting resource that needs that value
as a number:

```
// Proposal 2f: variant: specify merge-patch semantics with some variable replacement as an integer
{
    "spec": {
        // ...

        "supportTemplates": [{
            "replicationController": {
                "spec": {"replicas": 1}
            },
            // "$(REPLICAS)" is transformed into 5, not "5"
            "transform": [{"op": "add", "path": "/metadata/name", "value": "$(REPLICAS)", type: "number"}]
        }]
    }
}

```



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
fashion via DNS by leveraging information written to the endpoints by the endpoints controller. The
mechanism for that is either generic metadata delivered to the pod by the PetSet controller (an identity
field as in 1c, or an annotation as in 1b).

The end result might be DNS resolution as follows:

```
# service mongo pointing to pods created by PetSet mdb, with identities mdb-1, mdb-2, mdb-3

dig mongodb.namespace.svc.cluster.local +short A
172.130.16.50

dig mdb-1.mongodb.namespace.svc.cluster.local +short A
# IP of pod created for mdb-1

dig mdb-2.mongodb.namespace.svc.cluster.local +short A
# IP of pod created for mdb-2

dig mdb-3.mongodb.namespace.svc.cluster.local +short A
# IP of pod created for mdb-3

# made up name, returns a set of CNAMES? or SRV records for each identity ???
dig mongodb.namespace.identity.cluster.local CNAME
mdb-1.mongodb.namespace.svc.cluster.local
mdb-2.mongodb.namespace.svc.cluster.local
mdb-3.mongodb.namespace.svc.cluster.local
```


### Preventing duplicate identities

The PetSet controller is expected to execute like other controllers, as a single writer.  However, when
considering designing for safety first, the possibility of the controller running concurrently cannot
be overlooked, and so it is important to ensure that duplicate pod identities are not achieved.

There are two mechanisms to acheive this at the current time. One is to leverage unique names for pods
that carry the identity of the pod - this prevents duplication because etcd 2 can guarantee single
key transactionality. The other is to use the status field of the PetSet to coordinate membership
information. It is possible to leverage both at this time, and encourage users to not assume pod
name is significant, but users are likely to take what they can get. A downside of using unique names
is that it complicates pre-warming of pods and pod migration - on the other hand, those are also
advanced use cases that might be better solved by another, more specialized controller (a
MigratablePetSet).

## Examples

OPEN: concrete example walkthrough with a DB

## Overlap with other proposals

* Jobs can be used to perform a run-once initialization of the cluster
* Init containers can be used to prime PVs and config with the identity of the pod.
* Templates and how fields are overriden in the resulting object should have broad alignment
* DaemonSet defines the core model for how new controllers sit alongside replication controller



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/multiple-schedulers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
