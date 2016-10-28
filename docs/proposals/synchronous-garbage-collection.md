**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Overview](#overview)
- [API Design](#api-design)
  - [Standard Finalizers](#standard-finalizers)
  - [OwnerReference](#ownerreference)
  - [DeleteOptions](#deleteoptions)
- [Components changes](#components-changes)
  - [API Server](#api-server)
  - [Garbage Collector](#garbage-collector)
  - [Controllers](#controllers)
- [Handling circular dependencies](#handling-circular-dependencies)
- [Unhandled cases](#unhandled-cases)
- [Implications to existing clients](#implications-to-existing-clients)

<!-- END MUNGE: GENERATED_TOC -->

# Overview

Users of the server-side garbage collection need to determine if the garbage collection is done. For example:
* Currently `kubectl delete rc` blocks until all the pods are terminating. To convert to use server-side garbage collection, kubectl has to be able to determine if the garbage collection is done.
* [#19701](https://github.com/kubernetes/kubernetes/issues/19701#issuecomment-236997077) is a use case where the user needs to wait for all service dependencies garbage collected and their names released, before she recreates the dependencies.

We define the garbage collection as "done" when all the dependents are deleted from the key-value store, rather than merely in the terminating state. There are two reasons: *i)* for `Pod`s, the most usual garbage, only when they are deleted from the key-value store, we know kubelet has released resources they occupy; *ii)* some users need to recreate objects with the same names, they need to wait for the old objects to be deleted from the key-value store. (This limitation is because we index objects by their names in the key-value store today.)

Synchronous Garbage Collection is a best-effort (see [unhandled cases](#unhandled-cases)) mechanism that allows user to determine if the garbage collection is done: after the API server receives a deletion request of an owning object, the object keeps existing in the key-value store until all its dependents are deleted from the key-value store by the garbage collector.

Tracking issue: https://github.com/kubernetes/kubernetes/issues/29891

# API Design

## Standard Finalizers

We will introduce a new standard finalizer:

```go
const GCFinalizer string = “DeletingDependents”
```

This finalizer indicates the object is terminating and is waiting for its dependents whose `OwnerReference.BlockOwnerDeletion` is true get deleted.

## OwnerReference

```go
OwnerReference {
     ...
     // If true, AND if the owner has the "DeletingDependents" finalizer, then the owner cannot be deleted from the key-value store until this reference is removed.
     // Defaults to false.
     // To set this field, a user needs "delete" permission of the owner, otherwise 422 (Unprocessable Entity) will be returned.
     BlockOwnerDeletion *bool
}
```

The initial draft of the proposal did not include this field and it had a security loophole: a user who is only authorized to update one resource can set ownerReference to block the synchronous GC of other resources. Requiring users to explicitly set `BlockOwnerDeletion` allows the master to properly authorize the request.

## DeleteOptions

```go
DeleteOptions {
  …
  // Whether and how garbage collection will be performed.
  // Defaults to DeletePropagationDefault
  // Either this field or OrphanDependents may be set, but not both.
  PropagationPolicy *DeletePropagationPolicy
}

type DeletePropagationPolicy string

const (
    // The default depends on the existing finalizers on the object and the type of the object.
    DeletePropagationDefault DeletePropagationPolicy = "DeletePropagationDefault"
    // Orphans the dependents
    DeletePropagationOrphan DeletePropagationPolicy = "DeletePropagationOrphan"
    // Deletes the object from the key-value store, the garbage collector will delete the dependents in the background.
    DeletePropagationBackground DeletePropagationPolicy = "DeletePropagationBackground"
    // The object exists in the key-value store until the garbage collector deletes all the dependents whose ownerReference.blockOwnerDeletion=true from the key-value store.
    // API sever will put the "DeletingDependents" finalizer on the object, and sets its deletionTimestamp.
    // This policy is cascading, i.e., the dependents will be deleted with GarbageCollectionSynchronous.
    DeletePropagationForeground DeletePropagationPolicy = "DeletePropagationForeground"
)
```

The `DeletePropagationForeground` policy represents the synchronous GC mode.

`DeleteOptions.OrphanDependents *bool` will be marked as deprecated and will be removed in 1.7. Validation code will make sure only one of `OrphanDependents` and `PropagationPolicy` may be set. We decided not to add another `DeleteAfterDependentsDeleted *bool`, because together with `OrphanDependents`, it will result in 9 possible combinations and is thus confusing.

The conversion rules are described in the following table:

| 1.5                                      | pre 1.4/1.4              |
|------------------------------------------|--------------------------|
| DeletePropagationDefault                 | OrphanDependents==nil    |
| DeletePropagationOrphan                  | *OrphanDependents==true  |
| DeletePropagationBackground              | *OrphanDependents==false |
| DeletePropagationForeground              | N/A                      |

# Components changes

## API Server

`Delete()` function checks `DeleteOptions.PropagationPolicy`. If the policy is `DeletePropagationForeground`, the API server will update the object instead of deleting it, add the "DeletingDependents" finalizer, remove the "OrphanDependents" finalizer if it's present, and set the `ObjectMeta.DeletionTimestamp`.

When validating the ownerReference, API server needs to query the `Authorizer` to check if the user has "delete" permission of the owner object. It returns 422 if the user does not have the permissions but intends to set `OwnerReference.BlockOwnerDeletion` to true.

## Garbage Collector

**Modifications to processEvent()**

Currently `processEvent()` manages GC’s internal owner-dependency relationship graph, `uidToNode`. It updates `uidToNode` according to the Add/Update/Delete events in the cluster. To support synchronous GC, it has to:

* handle Add or Update events where `obj.Finalizers.Has(GCFinalizer) && obj.DeletionTimestamp != nil`. The object will be added into the `dirtyQueue`. The object will be marked as “GC in progress” in `uidToNode`.
* Upon receiving the deletion event of an object, put its owner into the `dirtyQueue` if the owner node is marked as "GC in progress". This is to force the `processItem()` (described next) to re-check if all dependents of the owner is deleted.

**Modifications to processItem()**

Currently `processItem()` consumes the `dirtyQueue`, requests the API server to delete an item if all of its owners do not exist. To support synchronous GC, it has to:

* treat an owner as "not exist" if `owner.DeletionTimestamp != nil && !owner.Finalizers.Has(OrphanFinalizer)`, otherwise synchronous GC will not progress because the owner keeps existing in the key-value store.
* when deleting dependents, if the owner's finalizers include `DeletingDependents`, it should use the `GarbageCollectionSynchronous` as GC policy.
* if an object has multiple owners, some owners still exist while other owners are in the synchronous GC stage, then according to the existing logic of GC, the object wouldn't be deleted. To unblock the synchronous GC of owners, `processItem()` has to remove the ownerReferences pointing to them.

In addition, if an object popped from `dirtyQueue` is marked as "GC in progress", `processItem()` treats it specially:

* To avoid racing with another controller, it requeues the object if `observedGeneration < Generation`. This is best-effort, see [unhandled cases](#unhandled-cases).
* Checks if the object has dependents
  * If not, send a PUT request to remove the `GCFinalizer`;
  * If so, then add all dependents to the `dirtryQueue`; we need bookkeeping to avoid adding the dependents repeatedly if the owner gets in the `synchronousGC queue` multiple times.

## Controllers

To utilize the synchronous garbage collection feature, controllers (e.g., the replicaset controller) need to set `OwnerReference.BlockOwnerDeletion` when creating dependent objects (e.g. pods).

# Handling circular dependencies

SynchronousGC will enter a deadlock in the presence of circular dependencies. The garbage collector can break the circle by lazily breaking circular dependencies: when `processItem()` processes an object, if it finds the object and all of its owners have the `GCFinalizer`, it removes the `GCFinalizer` from the object.

Note that the approach is not rigorous and thus having false positives. For example, if a user first sends a SynchronousGC delete request for an object, then sends the delete request for its owner, then `processItem()` will be fooled to believe there is a circle. We expect user not to do this. We can make the circle detection more rigorous if needed.

Circular dependencies are regarded as user error. If needed, we can add more guarantees to handle such cases later.

# Unhandled cases

* If the GC observes the owning object with the `GCFinalizer` before it observes the creation of all the dependents, GC will remove the finalizer from the owning object before all dependents are gone. Hence, synchronous GC is best-effort, though we guarantee that the dependents will be deleted eventually. We face a similar case when handling OrphanFinalizer, see [GC known issues](https://github.com/kubernetes/kubernetes/issues/26120).

# Implications to existing clients

Finalizer breaks an assumption that many Kubernetes components have: a deletion request with `grace period=0` will immediately remove the object from the key-value store. This is not true if an object has pending finalizers, the object will continue to exist, and currently the API server will not return an error in this case.

**Namespace controller** suffered from this [problem](https://github.com/kubernetes/kubernetes/issues/32519) and was fixed in [#32524](https://github.com/kubernetes/kubernetes/pull/32524) by retrying every 15s if there are objects with pending finalizers to be removed from the key-value store. Object with pending `GCFinalizer` might take arbitrary long time be deleted, so namespace deletion might time out.

**kubelet** deletes the pod from the key-value store after all its containers are terminated ([code](../../pkg/kubelet/status/status_manager.go#L441-L443)). It also assumes that if the API server does not return an error, the pod is removed from the key-value store. Breaking the assumption will not break `kubelet` though, because the `pod` must have already been in the terminated phase, `kubelet` will not care to manage it.

**Node controller** forcefully deletes pod if the pod is scheduled to a node that does not exist ([code](../../pkg/controller/node/nodecontroller.go#L474)). The pod will continue to exist if it has pending finalizers. The node controller will futilely retry the deletion. Also, the `node controller` forcefully deletes pods before deleting the node ([code](../../pkg/controller/node/nodecontroller.go#L592)). If the pods have pending finalizers, the `node controller` will go ahead deleting the node, leaving those pods behind. These pods will be deleted from the key-value store when the pending finalizers are removed.

**Podgc** deletes terminated pods if there are too many of them in the cluster. We need to make sure finalizers on Pods are taken off quickly enough so that the progress of `Podgc` is not affected.

**Deployment controller** adopts existing `ReplicaSet` (RS) if its template matches. If a matching RS has a pending `GCFinalizer`, deployment should adopt it, take its pods into account, but shouldn't try to mutate it, because the RS controller will ignore a RS that's being deleted. Hence, `deployment controller` should wait for the RS to be deleted, and then create a new one.

**Replication controller manager**, **Job controller**, and **ReplicaSet controller** ignore pods in terminated phase, so pods with pending finalizers will not block these controllers.

**StatefulSet controller** will be blocked by a pod with pending finalizers, so synchronous GC might slow down its progress.

**kubectl**: synchronous GC can simplify the **kubectl delete** reapers. Let's take the `deployment reaper` as an example, since it's the most complicated one. Currently, the reaper finds all `RS` with matching labels, scales them down, polls until `RS.Status.Replica` reaches 0, deletes the `RS`es, and finally deletes the `deployment`. If using synchronous GC, `kubectl delete deployment` is as easy as sending a synchronous GC delete request for the deployment, and polls until the deployment is deleted from the key-value store.

Note that this **changes the behavior** of `kubectl delete`. The command will be blocked until all pods are deleted from the key-value store, instead of being blocked until pods are in the terminating state. This means `kubectl delete` blocks for longer time, but it has the benefit that the resources used by the pods are released when the `kubectl delete` returns. To allow kubectl user not waiting for the cleanup, we will add a `--wait` flag. It defaults to true; if it's set to `false`, `kubectl delete` will send the delete request with `PropagationPolicy=DeletePropagationBackground` and return immediately.

To make the new kubectl compatible with the 1.4 and earlier masters, kubectl needs to switch to use the old reaper logic if it finds synchronous GC is not supported by the master.

1.4 `kubectl delete rc/rs` uses `DeleteOptions.OrphanDependents=true`, which is going to be converted to `DeletePropagationBackground` (see [API Design](#api-changes)) by a 1.5 master, so its behavior keeps the same.

Pre 1.4 `kubectl delete` uses `DeleteOptions.OrphanDependents=nil`, so does the 1.4 `kubectl delete` for resources other than rc and rs. The option is going to be converted to `DeletePropagationDefault` (see [API Design](#api-changes)) by a 1.5 master, so these commands behave the same as when working with a 1.4 master.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/synchronous-garbage-collection.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
