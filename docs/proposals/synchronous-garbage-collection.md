# Overview

Some users of the server-side garbage collection need to tell if the garbage collection is done. Synchronous Garbage Collection is a best-effort (see [unhandled cases](#unhandled-cases)) mechanism to enable such use cases: after the API server receives a deletion request of an owning object, the object keeps existing in the key-value store until all its dependents are deleted by the garbage collector.

Tracking issue: https://github.com/kubernetes/kubernetes/issues/29891

# Required code modification

We need to make changes in the API, the API Server, and the garbage collector to support synchronous garbage collection.

## API changes
```go
DeleteOptions {
  …
  // If SynchronousGarbageCollection is set, the object will not be deleted immediately. Instead, a GarbageCollectionInProgress finalizer will be placed on the object. The garbage collector will remove the finalizer from the object when all depdendents are deleted.
  // SynchronousGarbageCollection is ignored if OrphanDependents is true or nil.
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

* Needs to handle Add or Update events where `obj.Finalizers.Has(GCFinalizer) && obj.DeletionTimestamp != nil`. The object will be added into the `synchronousGC queue`. The object will be marked as “GC in progress” in GC’s internal owner-dependency relationship graph, `uidToNode`.
* Upon receiving the deletion event of an object, put its owner into the `synchronousGC queue`. This is to force the `GCFinalizer` (described next) to re-check if all dependents of the owner is deleted.

**Addition of GCFinalizer() routine**

* Pops an object from the `synchronousGC queue`.
* Ignores the object if it doesn’t exist in `uidToNode`, or if the object is not marked as “GC in progress” in `uidToNode`.
* To avoid racing with another controller, it requeues the object if `observedGeneration < Generation`. This is best-effort, see [unhandled cases](#unhandled-cases).
* Checks if the object has dependents
  * If not, send a PUT request to remove the GCFinalizer
  * If so, then add all dependents to the dirtryQueue; we need bookkeeping to avoid adding the dependents repeatedly if the owner gets in the `synchronousGC queue` multiple times. 

**Modifications to processItem()**

* Treat owner whose DeletionTimestamp is non-nil as “not exist”.
* When deleting dependents, it should use the same `DeleteOptions.SynchronousGC` as the owner’s finalizers suggest.

**Note**
We cannot have circular dependencies anymore. Otherwise SynchronousGC will enter a deadlock.

## Unhandled cases
* If the GC observes the owning object with the GCFinalizer before it observes the creation of all the dependents, GC will remove the finalizer from the owning object before all dependents are gone. Hence, “Synchronous GC” is best-effort, though we guarantee that the dependents will be deleted eventually.
