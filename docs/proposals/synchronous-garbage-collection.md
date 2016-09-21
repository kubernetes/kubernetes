<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Overview

Some users of the server-side garbage collection need to tell if the garbage collection is done ([example](https://github.com/kubernetes/kubernetes/issues/19701#issuecomment-236997077)). Synchronous Garbage Collection is a best-effort (see [unhandled cases](#unhandled-cases)) mechanism to enable such use cases: after the API server receives a deletion request of an owning object, the object keeps existing in the key-value store until all its dependents are deleted from the key-value store by the garbage collector.

Tracking issue: https://github.com/kubernetes/kubernetes/issues/29891

# Required code modification

We need to make changes in the API, the API Server, and the garbage collector to support synchronous garbage collection.

## API changes

```go
DeleteOptions {
  …
  // If SynchronousGarbageCollection is set, the object will not be deleted immediately. Instead, a GarbageCollectionInProgress finalizer will be placed on the object. The garbage collector will remove the finalizer from the object when all depdendents are deleted.
  // SynchronousGarbageCollection and OrphanDependents are exclusive.
  // SynchronousGarbageCollection default to false.
  // SynchronousGarbageCollection is cascading, i.e., the object’s dependents will be deleted with the same SynchronousGarbageCollection.
  SynchronousGarbageCollection *bool
}
```

We will introduce a new standard finalizer: const GCFinalizer string = “GarbageCollectionInProgress”

## Components changes

### API Server

Delete() function needs to check the DeleteOptions.SynchronousGarbageCollection.

* The option is ignored if DeleteOptions.OrphanDependents is true or nil.
* If the option is set, the API server will update the object instead of deleting it, add the finalizer, and set the `ObjectMeta.DeletionTimestamp`.

### Garbage Collector

**Modifications to processEvent()**

`processEvent()` manages GC’s internal owner-dependency relationship graph, `uidToNode`. It updates `uidToNode` according to the Add/Update/Delete events in the cluster. To support synchronous GC, it has to:

* handle Add or Update events where `obj.Finalizers.Has(GCFinalizer) && obj.DeletionTimestamp != nil`. The object will be added into the `synchronousGC queue`. The object will be marked as “GC in progress” in `uidToNode`.
* Upon receiving the deletion event of an object, put its owner into the `synchronousGC queue`. This is to force the `GCFinalizer` (described next) to re-check if all dependents of the owner is deleted.

**Addition of GCFinalizer() routine**

* Pops an object from the `synchronousGC queue`.
* Ignores the object if it doesn’t exist in `uidToNode`, or if the object is not marked as “GC in progress” in `uidToNode`.
* To avoid racing with another controller, it requeues the object if `observedGeneration < Generation`. This is best-effort, see [unhandled cases](#unhandled-cases).
* Checks if the object has dependents
  * If not, send a PUT request to remove the `GCFinalizer`
  * If so, then add all dependents to the `dirtryQueue`; we need bookkeeping to avoid adding the dependents repeatedly if the owner gets in the `synchronousGC queue` multiple times.

**Modifications to processItem()**

`processItem()` consumes the `dirtyQueue`, requests the API server to delete an item if all of its owners do not exist. To support synchronous GC, it has to:

* treat an owner as "not exist" if `owner.DeletionTimestamp != nil && !owner.Finalizers.Has(OrphanFinalizer)`, otherwise Synchronous GC will not progress because the owner keeps existing in the key-value store.
* when deleting dependents, it should use the same `DeleteOptions.SynchronousGC` as the owner’s finalizers suggest.
* if an object has multiple owners, some owners still exit while other owners are in the synchronous GC stage, then according to the existing logic of GC, the object wouldn't be deleted. To unblock the synchronous GC of those owners, `processItem()` has to remove the ownerReferences pointing to them.

**Handling circular dependencies**

SynchronousGC will enter a deadlock in the presence of circular dependencies. The garbage collector can break the circle by lazily detecting circular dependencies: when `processItem()` processes an object, if it finds the object and all of its owners have the `GCFinalizer`, it searches the internal owner-dependency relationship graph (`uidToNode`) to check if the object and any of its owner are in a circle where all objects have the `GCFinalizer`. If so, it removes the `GCFinzlier` from the object to break the circle.

## Unhandled cases

* If the GC observes the owning object with the `GCFinalizer` before it observes the creation of all the dependents, GC will remove the finalizer from the owning object before all dependents are gone. Hence, “Synchronous GC” is best-effort, though we guarantee that the dependents will be deleted eventually. We face a similar case when handling OrphanFinalizer, see [GC known issues](https://github.com/kubernetes/kubernetes/issues/26120).


## Implications to existing clients

Finalizer breaks an assumption that many Kubernetes components have: a deletion request with `grace period=0` will immediately remove the object from the key-value store. This is not true if an object has pending finalizers, the object will continue to exist, and currently the API server will not return an error in this case.

**Namespace controller** suffered from this [problem](https://github.com/kubernetes/kubernetes/issues/32519) and was fixed in [#32524](https://github.com/kubernetes/kubernetes/pull/32524) by retrying every 15s if there are objects with pending finalizers to be removed from the key-value store. Object with pending `GCFinalizer` might take arbitrary long time be deleted, so namespace deletion might time out.

**kubelet** deletes the pod from the key-value store after all its containers are terminated ([code](../../pkg/kubelet/status/status_manager.go#L441-L443)). It also assumes that if the API server does not return an error, the pod is removed from the key-value store. Breaking the assumption will not break `kubelet` though, because the `pod` must have already been in the terminated `phase`, `kubelet` will not care to manage it.

**Node controller** forcefully deletes pod if the pod is scheduled to a node that does not exist ([code](../../pkg/controller/node/nodecontroller.go#L474)). The pod will continue to exist if it has pending finalizers. The node controller will futilely retry the deletion. Also, the `node controller` forcefully deletes pods before deleting the node ([code](../../pkg/controller/node/nodecontroller.go#L592)). If the pods have pending finalizers, the `node controller` will go ahead deleting the node, leaving those pods behind. These pods will be deleted from the key-value store when the pending finalizers are removed.

**Podgc** deletes terminated pods if there are too many of them in the cluster. `Podgc` should remove any pending finalizers to make sure the pods are deleted.

**Deployment controller** adopts existing `ReplicaSet` (RS) if its template matches. If a matching RS has a pending `GCFinalizer`, deployment shouldn't adopt it, because the RS controller will not scale up/down a RS that's being deleted. Hence, `deployment controller` needs to check if a RS is being deleted before adopting it. If the RS is being deleted, then the `deployment controller` should wait for the status of the RS to show 0 replicas to avoid creating extra pods, then create a new RS.

**Replication controller manager**, **Job controller**, and **ReplicaSet controller** ignore pods in terminated phase, so pods with pending finalizers will not block these controllers.

**PetSet controller** will be blocked by a pod with pending finalizers, so Synchronous GC might slow down its progress.

**kubectl**: synchronous GC can replace the **kubectl delete** reapers. Currently `kubectl delete` blocks until all dependents and the owner are deleted. To maintain this behavior, after switched to using synchronous GC, *kubectl delete* needs to poll on the removal of the owner object.

## Security implications

A user who is authorized to update one object can affect the synchronous GC behavior of another object. Specifically, by setting an object as a pod's owner, and setting a very long grace termination period for the pod, a user can make the synchronous GC of the owner to take long time.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/synchronous-garbage-collection.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
