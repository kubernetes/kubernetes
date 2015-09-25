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
[here](http://releases.k8s.io/release-1.0/docs/design/namespaces.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Namespaces

## Abstract

A Namespace is a mechanism to partition resources created by users into
a logically named group.

## Motivation

A single cluster should be able to satisfy the needs of multiple user communities.

Each user community wants to be able to work in isolation from other communities.

Each user community has its own:

1. resources (pods, services, replication controllers, etc.)
2. policies (who can or cannot perform actions in their community)
3. constraints (this community is allowed this much quota, etc.)

A cluster operator may create a Namespace for each unique user community.

The Namespace provides a unique scope for:

1. named resources (to avoid basic naming collisions)
2. delegated management authority to trusted users
3. ability to limit community resource consumption

## Use cases

1.  As a cluster operator, I want to support multiple user communities on a single cluster.
2.  As a cluster operator, I want to delegate authority to partitions of the cluster to trusted users
    in those communities.
3.  As a cluster operator, I want to limit the amount of resources each community can consume in order
    to limit the impact to other communities using the cluster.
4.  As a cluster user, I want to interact with resources that are pertinent to my user community in
    isolation of what other user communities are doing on the cluster.

## Design

### Data Model

A *Namespace* defines a logically named group for multiple *Kind*s of resources.

```go
type Namespace struct {
  TypeMeta   `json:",inline"`
  ObjectMeta `json:"metadata,omitempty"`

  Spec NamespaceSpec `json:"spec,omitempty"`
  Status NamespaceStatus `json:"status,omitempty"`
}
```

A *Namespace* name is a DNS compatible label.

A *Namespace* must exist prior to associating content with it.

A *Namespace* must not be deleted if there is content associated with it.

To associate a resource with a *Namespace* the following conditions must be satisfied:

1.  The resource's *Kind* must be registered as having *RESTScopeNamespace* with the server
2.  The resource's *TypeMeta.Namespace* field must have a value that references an existing *Namespace*

The *Name* of a resource associated with a *Namespace* is unique to that *Kind* in that *Namespace*.

It is intended to be used in resource URLs; provided by clients at creation time, and encouraged to be
human friendly; intended to facilitate idempotent creation, space-uniqueness of singleton objects,
distinguish distinct entities, and reference particular entities across operations.

### Authorization

A *Namespace* provides an authorization scope for accessing content associated with the *Namespace*.

See [Authorization plugins](../admin/authorization.md)

### Limit Resource Consumption

A *Namespace* provides a scope to limit resource consumption.

A *LimitRange* defines min/max constraints on the amount of resources a single entity can consume in
a *Namespace*.

See [Admission control: Limit Range](admission_control_limit_range.md)

A *ResourceQuota* tracks aggregate usage of resources in the *Namespace* and allows cluster operators
to define *Hard* resource usage limits that a *Namespace* may consume.

See [Admission control: Resource Quota](admission_control_resource_quota.md)

### Finalizers

Upon creation of a *Namespace*, the creator may provide a list of *Finalizer* objects.

```go
type FinalizerName string

// These are internal finalizers to Kubernetes, must be qualified name unless defined here
const (
  FinalizerKubernetes FinalizerName = "kubernetes"
)

// NamespaceSpec describes the attributes on a Namespace
type NamespaceSpec struct {
  // Finalizers is an opaque list of values that must be empty to permanently remove object from storage
  Finalizers []FinalizerName
}
```

A *FinalizerName* is a qualified name.

The API Server enforces that a *Namespace* can only be deleted from storage if and only if
it's *Namespace.Spec.Finalizers* is empty.

A *finalize* operation is the only mechanism to modify the *Namespace.Spec.Finalizers* field post creation.

Each *Namespace* created has *kubernetes* as an item in its list of initial *Namespace.Spec.Finalizers*
set by default.

### Phases

A *Namespace* may exist in the following phases.

```go
type NamespacePhase string
const(
  NamespaceActive NamespacePhase = "Active"
  NamespaceTerminating NamespaceTerminating = "Terminating"
)

type NamespaceStatus struct { 
  ...
  Phase NamespacePhase 
}
```

A *Namespace* is in the **Active** phase if it does not have a *ObjectMeta.DeletionTimestamp*.

A *Namespace* is in the **Terminating** phase if it has a *ObjectMeta.DeletionTimestamp*.

**Active**

Upon creation, a *Namespace* goes in the *Active* phase.  This means that content may be associated with
a namespace, and all normal interactions with the namespace are allowed to occur in the cluster.

If a DELETE request occurs for a *Namespace*, the *Namespace.ObjectMeta.DeletionTimestamp* is set
to the current server time.  A *namespace controller* observes the change, and sets the *Namespace.Status.Phase*
to *Terminating*.

**Terminating**

A *namespace controller* watches for *Namespace* objects that have a *Namespace.ObjectMeta.DeletionTimestamp*
value set in order to know when to initiate graceful termination of the *Namespace* associated content that
are known to the cluster.

The *namespace controller* enumerates each known resource type in that namespace and deletes it one by one.

Admission control blocks creation of new resources in that namespace in order to prevent a race-condition
where the controller could believe all of a given resource type had been deleted from the namespace,
when in fact some other rogue client agent had created new objects.  Using admission control in this
scenario allows each of registry implementations for the individual objects to not need to take into account Namespace life-cycle.

Once all objects known to the *namespace controller* have been deleted, the *namespace controller*
executes a *finalize* operation on the namespace that removes the *kubernetes* value from
the *Namespace.Spec.Finalizers* list.

If the *namespace controller* sees a *Namespace* whose *ObjectMeta.DeletionTimestamp* is set, and
whose *Namespace.Spec.Finalizers* list is empty, it will signal the server to permanently remove
the *Namespace* from storage by sending a final DELETE action to the API server.

### REST API

To interact with the Namespace API:

| Action | HTTP Verb | Path | Description |
| ------ | --------- | ---- | ----------- |
| CREATE | POST | /api/{version}/namespaces | Create a namespace |
| LIST | GET | /api/{version}/namespaces | List all namespaces |
| UPDATE | PUT | /api/{version}/namespaces/{namespace} | Update namespace {namespace} |
| DELETE | DELETE | /api/{version}/namespaces/{namespace} | Delete namespace {namespace} |
| FINALIZE | POST | /api/{version}/namespaces/{namespace}/finalize | Finalize namespace {namespace} |
| WATCH | GET | /api/{version}/watch/namespaces | Watch all namespaces |

This specification reserves the name *finalize* as a sub-resource to namespace.

As a consequence, it is invalid to have a *resourceType* managed by a namespace whose kind is *finalize*.

To interact with content associated with a Namespace:

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/namespaces/{namespace}/{resourceType}/ | Create instance of {resourceType} in namespace {namespace} |
| GET | GET | /api/{version}/namespaces/{namespace}/{resourceType}/{name} | Get instance of {resourceType} in namespace {namespace} with {name} |
| UPDATE | PUT | /api/{version}/namespaces/{namespace}/{resourceType}/{name} | Update instance of {resourceType} in namespace {namespace} with {name} |
| DELETE | DELETE | /api/{version}/namespaces/{namespace}/{resourceType}/{name} | Delete instance of {resourceType} in namespace {namespace} with {name} |
| LIST | GET | /api/{version}/namespaces/{namespace}/{resourceType} | List instances of {resourceType} in namespace {namespace} |
| WATCH | GET | /api/{version}/watch/namespaces/{namespace}/{resourceType} | Watch for changes to a {resourceType} in namespace {namespace} |
| WATCH | GET | /api/{version}/watch/{resourceType} | Watch for changes to a {resourceType} across all namespaces |
| LIST | GET | /api/{version}/list/{resourceType} | List instances of {resourceType} across all namespaces |

The API server verifies the *Namespace* on resource creation matches the *{namespace}* on the path.

The API server will associate a resource with a *Namespace* if not populated by the end-user based on the *Namespace* context
of the incoming request.  If the *Namespace* of the resource being created, or updated does not match the *Namespace* on the request,
then the API server will reject the request.

### Storage

A namespace provides a unique identifier space and therefore must be in the storage path of a resource.

In etcd, we want to continue to still support efficient WATCH across namespaces.

Resources that persist content in etcd will have storage paths as follows:

/{k8s_storage_prefix}/{resourceType}/{resource.Namespace}/{resource.Name}

This enables consumers to WATCH /registry/{resourceType} for changes across namespace of a particular {resourceType}.

### Kubelet

The kubelet will register pod's it sources from a file or http source with a namespace associated with the
*cluster-id*

### Example: OpenShift Origin managing a Kubernetes Namespace

In this example, we demonstrate how the design allows for agents built on-top of
Kubernetes that manage their own set of resource types associated with a *Namespace*
to take part in Namespace termination.

OpenShift creates a Namespace in Kubernetes

```json
{
  "apiVersion":"v1",
  "kind": "Namespace",
  "metadata": {
    "name": "development",
    "labels": {
      "name": "development"
    }
  },
  "spec": {
    "finalizers": ["openshift.com/origin", "kubernetes"]
  },
  "status": {
    "phase": "Active"
  }
}
```

OpenShift then goes and creates a set of resources (pods, services, etc) associated
with the "development" namespace.  It also creates its own set of resources in its
own storage associated with the "development" namespace unknown to Kubernetes.

User deletes the Namespace in Kubernetes, and Namespace now has following state:

```json
{
  "apiVersion":"v1",
  "kind": "Namespace",
  "metadata": {
    "name": "development",
    "deletionTimestamp": "..."
    "labels": {
      "name": "development"
    }
  },
  "spec": {
    "finalizers": ["openshift.com/origin", "kubernetes"]
  },
  "status": {
    "phase": "Terminating"
  }
}
```

The Kubernetes *namespace controller* observes the namespace has a *deletionTimestamp*
and begins to terminate all of the content in the namespace that it knows about.  Upon
success, it executes a *finalize* action that modifies the *Namespace* by
removing *kubernetes* from the list of finalizers:

```json
{
  "apiVersion":"v1",
  "kind": "Namespace",
  "metadata": {
    "name": "development",
    "deletionTimestamp": "..."
    "labels": {
      "name": "development"
    }
  },
  "spec": {
    "finalizers": ["openshift.com/origin"]
  },
  "status": {
    "phase": "Terminating"
  }
}
```

OpenShift Origin has its own *namespace controller* that is observing cluster state, and
it observes the same namespace had a *deletionTimestamp* assigned to it.  It too will go
and purge resources from its own storage that it manages associated with that namespace.
Upon completion, it executes a *finalize* action and removes the reference to "openshift.com/origin"
from the list of finalizers.

This results in the following state:

```json
{
  "apiVersion":"v1",
  "kind": "Namespace",
  "metadata": {
    "name": "development",
    "deletionTimestamp": "..."
    "labels": {
      "name": "development"
    }
  },
  "spec": {
    "finalizers": []
  },
  "status": {
    "phase": "Terminating"
  }
}
```

At this point, the Kubernetes *namespace controller* in its sync loop will see that the namespace
has a deletion timestamp and that its list of finalizers is empty.  As a result, it knows all
content associated from that namespace has been purged.  It performs a final DELETE action
to remove that Namespace from the storage.

At this point, all content associated with that Namespace, and the Namespace itself are gone.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/namespaces.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
