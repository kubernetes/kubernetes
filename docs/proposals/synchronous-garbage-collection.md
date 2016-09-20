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

SynchronousGC will enter a deadlock in the presence of circular dependencies. Here are two alternative approaches to break the deadlock:

1. Timeout a `GCFinalizer  `: To implement the timeout, GC adds an object that has a `GCFinalizer` into a [delaying queue](../../pkg/util/workqueue/delaying_queue.go) when it's observed, and removes the `GCFinalizer` from it when the time is up. The timeout value should be proportional to the number of dependents, including indirect ones.

2. Lazily detecting circular dependencies: when `processItem()` processes an object, if it finds the object and all of its owners have the `GCFinalizer`, it searches the internal owner-dependency relationship graph (`uidToNode`) to check if the object and any of its owner are in a circle where all objects have the `GCFinalizer`. If so, it removes the `GCFinzlier` from the object to break the circle.

## Unhandled cases
* If the GC observes the owning object with the `GCFinalizer` before it observes the creation of all the dependents, GC will remove the finalizer from the owning object before all dependents are gone. Hence, “Synchronous GC” is best-effort, though we guarantee that the dependents will be deleted eventually. We face a similar case when handling OrphanFinalizer, see [GC known issues](https://github.com/kubernetes/kubernetes/issues/26120).


## Implications to existing clients

Finalizer breaks an assumption that many Kubernetes components have: a deletion request with `grace period=0` will immediately remove the object from the key-value store. This is not true if an object has pending finalizers, the object will continue to exist, and currently the API server will not return an error in this case.

**Namespce controller** suffered from this [problem](https://github.com/kubernetes/kubernetes/issues/32519) and was fixed in [#32524](https://github.com/kubernetes/kubernetes/pull/32524) by retrying every 15s if there are objects with pending finalizers to be removed from the key-value store. Object with pending `GCFinalizer` might take arbitrary long time be deleted, so namespace deletion might time out.

**kubelet** deletes the pod from the key-value store after all its containers are terminated ([code](https://github.com/kubernetes/kubernetes/blob/master/pkg/kubelet/status/status_manager.go#L441-L443)). It also assumes if API server does not return an error, the pod is removed from the key-value store. Breaking the assumption will not break `kubelet` though, because the `pod` must have already been in the terminated `phase`, `kubelet` will not care to manage it.

**Node controller** forcefully deletes pod if the pod is scheduled to a node that does not exist ([code](https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/node/nodecontroller.go#L474)). The pod will continue to exist if it has pending finalizers. The node controller will futilely retry the deletion. The `node controller` forcefully deletes pods before deleting the node ([code](https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/node/nodecontroller.go#L592)). If the pods have pending finalizers, the `node controller` will go ahead deleting the node, leaving those pods behind. Other components will take care of the pending finalizers. 

**Podgc** deletes terminated pods if there are too many of them in the cluster. `Podgc` should remove any pending finalizers to make sure the pods are deleted.

**Deployment controller** adopts existing `ReplicaSet` (RS) if its template matches. If a matching RS has a pending `GCFinalizer`, deployment shouldn't adopt it, because the RS controller will scale up/down a RS that's being deleted. Hence, `deployment controller` needs to check if a RS is being deleted before adopting it. If the RS is being deleted, then the `deployment controller` should wait for the status of the RS showing 0 replicas, and then create a new RS.

**Replication controller manager**, **Job controller**, and **ReplicaSet controller** send deletion request of pods. `kubelet` will drive these pods to the terminated phase, so the pods will be ignored by the controllers even if they keep existing in the key-value store because of pending finalizers.

**Endpoints controller** deletes endpoints. It does not double check if the endpoint is gone, so 

One usage of the synchronous GC is to replace the **kubectl delete** reapers. Currently `kubectl delete` blocks until all dependents and the owner are deleted. To maintain this behavior, after switched to using synchronous GC, *kubectl delete* needs to poll on the removal of the owner object.

## Security implications

A user who are authorized to update one object can affect the synchronous GC behavior of another object. Specifically, even if a user is only authorized to update a pod, he can set another object as the pod's owner, and set a very long grace termination period for the pod, then he makes the synchronous GC of the owner takes long time.
