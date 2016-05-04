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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

**Table of Contents**

- [Overview](#overview)
- [Cascading deletion with Garbage Collector](#cascading-deletion-with-garbage-collector)
- [Orphaning the descendants with "orphan" finalizer](#orphaning-the-descendants-with-orphan-finalizer)
  - [Part I. The finalizer framework](#part-i-the-finalizer-framework)
  - [Part II. The "orphan" finalizer](#part-ii-the-orphan-finalizer)
- [Related issues](#related-issues)
  - [Orphan adoption](#orphan-adoption)
  - [Upgrading a cluster to support cascading deletion](#upgrading-a-cluster-to-support-cascading-deletion)
- [End-to-End Examples](#end-to-end-examples)
  - [Life of a Deployment and its descendants](#life-of-a-deployment-and-its-descendants)
  - [Life of a ConfigMap](#life-of-a-configmap)
- [Open Questions](#open-questions)
- [Considered and Rejected Designs](#considered-and-rejected-designs)
- [1. Tombstone + GC](#1-tombstone--gc)
- [2. Recovering from abnormal cascading deletion](#2-recovering-from-abnormal-cascading-deletion)


# Overview

Currently most cascading deletion logic is implemented at client-side. For example, when deleting a replica set, kubectl uses a reaper to delete the created pods and then delete the replica set. We plan to move the cascading deletion to the server to simplify the client-side logic. In this proposal, we present the garbage collector which implements cascading deletion for all API resources in a generic way; we also present the finalizer framework, particularly the "orphan" finalizer, to enable flexible alternation between cascading deletion and orphaning.

Goals of the design include:
* Supporting cascading deletion at the server-side.
* Centralizing the cascading deletion logic, rather than spreading in controllers.
* Allowing optionally orphan the dependent objects.

Non-goals include:
* Releasing the name of an object immediately, so it can be reused ASAP.
* Propagating the grace period in cascading deletion.

# Cascading deletion with Garbage Collector

## API Changes

```
type ObjectMeta struct {
	...	
	OwnerReferences []OwnerReference
}
```

**ObjectMeta.OwnerReferences**:
List of objects depended by this object. If ***all*** objects in the list have been deleted, this object will be garbage collected. For example, a replica set `R` created by a deployment `D` should have an entry in ObjectMeta.OwnerReferences pointing to `D`, set by the deployment controller when `R` is created. This field can be updated by any client that has the privilege to both update ***and*** delete the object. For safety reasons, we can add validation rules to restrict what resources could be set as owners. For example, Events will likely be banned from being owners.

```
type OwnerReference struct {
	// Version of the referent.
	APIVersion string
	// Kind of the referent.
	Kind string
	// Name of the referent.
	Name string
	// UID of the referent.
	UID types.UID
}
```

**OwnerReference struct**: OwnerReference contains enough information to let you identify an owning object. Please refer to the inline comments for the meaning of each field. Currently, an owning object must be in the same namespace as the dependent object, so there is no namespace field.

## New components: the Garbage Collector

The Garbage Collector is responsible to delete an object if none of the owners listed in the object's OwnerReferences exist.
The Garbage Collector consists of a scanner, a garbage processor, and a propagator.
* Scanner:
  * Uses the discovery API to detect all the resources supported by the system.
  * Periodically scans all resources in the system and adds each object to the *Dirty Queue*.

* Garbage Processor:
  * Consists of the *Dirty Queue* and workers.
  * Each worker:
    * Dequeues an item from *Dirty Queue*.
    * If the item's OwnerReferences is empty, continues to process the next item in the *Dirty Queue*.
    * Otherwise checks each entry in the OwnerReferences:
      * If at least one owner exists, do nothing.
      * If none of the owners exist, requests the API server to delete the item.

* Propagator:
  * The Propagator is for optimization, not for correctness.
  * Consists of an *Event Queue*,  a single worker, and a DAG of owner-dependent relations.
    * The DAG stores only name/uid/orphan triplets, not the entire body of every item.
  * Watches for create/update/delete events for all resources, enqueues the events to the *Event Queue*.
  * Worker:
    * Dequeues an item from the *Event Queue*.
    * If the item is an creation or update, then updates the DAG accordingly.
      * If the object has an owner and the owner doesn’t exist in the DAG yet, then apart from adding the object to the DAG, also enqueues the object to the *Dirty Queue*.
    * If the item is a deletion, then removes the object from the DAG, and enqueues all its dependent objects to the *Dirty Queue*.
  * The propagator shouldn't need to do any RPCs, so a single worker should be sufficient. This makes locking easier.
  * With the Propagator, we *only* need to run the Scanner when starting the GC to populate the DAG and the *Dirty Queue*.

# Orphaning the descendants with "orphan" finalizer

Users may want to delete an owning object (e.g., a replicaset) while orphaning the dependent object (e.g., pods), that is, leaving the dependent objects untouched. We support such use cases by introducing the "orphan" finalizer. Finalizer is a generic API that has uses other than supporting orphaning, so we first describe the generic finalizer framework, then describe the specific design of the "orphan" finalizer.

## Part I. The finalizer framework

## API changes

```
type ObjectMeta struct {
	…
	Finalizers []string
}
```

**ObjectMeta.Finalizers**: List of finalizers that need to run before deleting the object. This list must be empty before the object is deleted from the registry. Each string in the list is an identifier for the responsible component that will remove the entry from the list. If the deletionTimestamp of the object is non-nil, entries in this list can only be removed. For safety reasons, updating finalizers requires special privileges. To enforce the admission rules, we will expose finalizers as a subresource and disallow directly changing finalizers when updating the main resource.

## New components

* Finalizers:
  * Like a controller, a finalizer is always running.
  * A third party can develop and run their own finalizer in the cluster. A finalizer doesn't need to be registered with the API server.
  * Watches for update events that meet two conditions:
    1. the updated object has the identifier of the finalizer in ObjectMeta.Finalizers;
    2. ObjectMeta.DeletionTimestamp is updated from nil to non-nil.
  * Applies the finalizing logic to the object in the update event.
  * After the finalizing logic is completed, removes itself from ObjectMeta.Finalizers.
  * The API server deletes the object after the last finalizer removes itself from the ObjectMeta.Finalizers field.
  * Because it's possible for the finalizing logic to be applied multiple times (e.g., the finalizer crashes after applying the finalizing logic but before being removed form ObjectMeta.Finalizers), the finalizing logic has to be idempotent.
  * If a finalizer fails to act in a timely manner, users with proper privileges can manually remove the finalizer from ObjectMeta.Finalizers. We will provide a kubectl command to do this.

## Changes to existing components

* API server:
  * Deletion handler:
    * If the `ObjectMeta.Finalizers` of the object being deleted is non-empty, then updates the DeletionTimestamp, but does not delete the object.
    * If the `ObjectMeta.Finalizers` is empty and the options.GracePeriod is zero, then deletes the object. If the options.GracePeriod is non-zero, then just updates the DeletionTimestamp.
  * Update handler:
    * If the update removes the last finalizer, and the DeletionTimestamp is non-nil, and the DeletionGracePeriodSeconds is zero, then deletes the object from the registry.
    * If the update removes the last finalizer, and the DeletionTimestamp is non-nil, but the DeletionGracePeriodSeconds is non-zero, then just updates the object.

## Part II. The "orphan" finalizer

## API changes

```
type DeleteOptions struct {
	…
	OrphanDependents bool
}
```

**DeleteOptions.OrphanDependents**: allows a user to express whether the dependent objects should be orphaned. It defaults to true, because controllers before release 1.2 expect dependent objects to be orphaned.

## Changes to existing components

* API server:
When handling a deletion request, depending on if DeleteOptions.OrphanDependents is true, the API server updates the object to add/remove the "orphan" finalizer to/from the ObjectMeta.Finalizers map.


## New components

Adding a fourth component to the Garbage Collector, the"orphan" finalizer:
* Watches for update events as described in [Part I](#part-i-the-finalizer-framework).
* Removes the object in the event from the `OwnerReferences` of its dependents.
  * dependent objects can be found via the DAG kept by the GC, or by relisting the dependent resource and checking the OwnerReferences field of each potential dependent object.
* Also removes any dangling owner references the dependent objects have.
* At last, removes the itself from the `ObjectMeta.Finalizers` of the object.

# Related issues

## Orphan adoption

Controllers are responsible for adopting orphaned dependent resources. To do so, controllers
* Checks a potential dependent object’s OwnerReferences to determine if it is orphaned.
* Fills the OwnerReferences if the object matches the controller’s selector and is orphaned.

There is a potential race between the "orphan" finalizer removing an owner reference and the controllers adding it back during adoption. Imagining this case: a user deletes an owning object and intends to orphan the dependent objects, so the GC removes the owner from the dependent object's OwnerReferences list, but the controller of the owner resource hasn't observed the deletion yet, so it adopts the dependent again and adds the reference back, resulting in the mistaken deletion of the dependent object. This race can be avoided by implementing Status.ObservedGeneration in all resources. Before updating the dependent Object's OwnerReferences, the "orphan" finalizer checks Status.ObservedGeneration of the owning object to ensure its controller has already observed the deletion.

## Upgrading a cluster to support cascading deletion

For the master, after upgrading to a version that supports cascading deletion, the OwnerReferences of existing objects remain empty, so the controllers will regard them as orphaned and start the adoption procedures. After the adoptions are done, server-side cascading will be effective for these existing objects.

For nodes, cascading deletion does not affect them.

For kubectl, we will keep the kubectl’s cascading deletion logic for one more release.

# End-to-End Examples

This section presents two examples of all components working together to enforce the cascading deletion or orphaning.

## Life of a Deployment and its descendants

1. User creates a deployment `D1`.
2. The Propagator of the GC observes the creation. It creates an entry of `D1` in the DAG.
3. The deployment controller observes the creation of `D1`. It creates the replicaset `R1`, whose OwnerReferences field contains a reference to `D1`, and has the "orphan" finalizer in its ObjectMeta.Finalizers map.
4. The Propagator of the GC observes the creation of `R1`. It creates an entry of `R1` in the DAG, with `D1` as its owner.
5. The replicaset controller observes the creation of `R1` and creates Pods `P1`~`Pn`, all with `R1` in their OwnerReferences.
6. The Propagator of the GC observes the creation of `P1`~`Pn`. It creates entries for them in the DAG, with `R1` as their owner.

  ***In case the user wants to cascadingly delete `D1`'s descendants, then***

7. The user deletes the deployment `D1`, with `DeleteOptions.OrphanDependents=false`. API server checks if `D1` has "orphan" finalizer in its Finalizers map, if so, it updates `D1` to remove the "orphan" finalizer. Then API server deletes `D1`.
8. The "orphan" finalizer does *not* take any action, because the observed deletion shows `D1` has an empty Finalizers map.
9. The Propagator of the GC observes the deletion of `D1`. It deletes `D1` from the DAG. It adds its dependent object, replicaset `R1`, to the *dirty queue*.
10. The Garbage Processor of the GC dequeues `R1` from the *dirty queue*. It finds `R1` has an owner reference pointing to `D1`, and `D1` no longer exists, so it requests API server to delete `R1`, with `DeleteOptions.OrphanDependents=false`. (The Garbage Processor should always set this field to false.)
11. The API server updates `R1` to remove the "orphan" finalizer if it's in the `R1`'s Finalizers map. Then the API server deletes `R1`, as `R1` has an empty Finalizers map.
12. The Propagator of the GC observes the deletion of `R1`. It deletes `R1` from the DAG. It adds its dependent objects, Pods `P1`~`Pn`, to the *Dirty Queue*.
13. The Garbage Processor of the GC dequeues `Px` (1 <= x <= n) from the *Dirty Queue*. It finds that `Px` have an owner reference pointing to `D1`, and `D1` no longer exists, so it requests API server to delete `Px`, with `DeleteOptions.OrphanDependents=false`.
14. API server deletes the Pods.

  ***In case the user wants to orphan `D1`'s descendants, then***

7. The user deletes the deployment `D1`, with `DeleteOptions.OrphanDependents=true`.
8. The API server first updates `D1`, with DeletionTimestamp=now and DeletionGracePeriodSeconds=0, increments the Generation by 1, and add the "orphan" finalizer to ObjectMeta.Finalizers if it's not present yet. The API server does not delete `D1`, because its Finalizers map is not empty.
9. The deployment controller observes the update, and acknowledges by updating the `D1`'s ObservedGeneration. The deployment controller won't create more replicasets on `D1`'s behalf.
10. The "orphan" finalizer observes the update, and notes down the Generation. It waits until the ObservedGeneration becomes equal to or greater than the noted Generation. Then it updates `R1` to remove `D1` from its OwnerReferences. At last, it updates `D1`, removing itself from `D1`'s Finalizers map.
11. The API server handles the update of `D1`, because *i)* DeletionTimestamp is non-nil, *ii)*  the DeletionGracePeriodSeconds is zero, and *iii)* the last finalizer is removed from the Finalizers map, API server deletes `D1`.
12. The Propagator of the GC observes the deletion of `D1`. It deletes `D1` from the DAG. It adds its dependent, replicaset `R1`, to the *Dirty Queue*.
13. The Garbage Processor of the GC dequeues `R1` from the *Dirty Queue* and skips it, because its OwnerReferences is empty.

## Life of a ConfigMap

1. User creates a ConfigMap `C1`.
2. User creates a Deployment `D1`, which refers `C1` in the pod template.
3. The deployment controller has observed the creation of `D1`. It creates replicaset `R1`, which also refer `C1` in the pod template. It then updates `C1`, adding `R1` to the OwnerReferences.
4. `R1` is deleted with DeleteOptions.OrphanDependents=false, either caused by the cascading deletion of `D1`, or a rolling update of `D1` where `R1` is removed from the deployment revision history.
5. API server handles the deletion of `R1`, it first removes the "orphan" finalizer from `R1`'s Finalizers map is it's present, then it deletes `R1` from the registry.
6. The Propagator of the GC observes the deletion of `R1`. It deletes `R1` from the DAG. It adds its dependent objects, including ConfigMap `C1`, to the *dirty queue*.
7. The Garbage Processor of the GC dequeues `C1` from the *dirty queue*. `C1` may have owner references to other replicasets. If none of its owners exist, the Garbage Processor requests API server to delete `C1`. Otherwise, it does nothing.

# Open Questions

1. In case an object has multiple owners, some owners are deleted with DeleteOptions.OrphanDependents=true, and some are deleted with DeleteOptions.OrphanDependents=false, what should happen to the object?

  The presented design will respect the setting in the deletion request of last owner.

2. How to propagate the grace period in a cascading deletion? For example, when deleting a ReplicaSet with grace period of 5s, a user may expect the same grace period to be applied to the deletion of the Pods controlled the ReplicaSet.

  Propagating grace period in a cascading deletion is a ***non-goal*** of this proposal. Nevertheless, the presented design can be extended to support it. A tentative solution is letting the garbage collector to propagate the grace period when deleting dependent object. To persist the grace period set by the user, the owning object should not be deleted from the registry until all its dependent objects are in the graceful deletion state. This could be ensured by introducing another finalizer, tentatively named as the "populating graceful deletion" finalizer. Upon receiving the graceful deletion request, the API server adds this finalizer to the finalizers list of the owning object. Later the GC will remove it when all dependents are in the graceful deletion state.

  [#25055](https://github.com/kubernetes/kubernetes/issues/25055) tracks this problem.

3. How can a client know when the cascading deletion is completed?

  A tentative solution is introducing a "completing cascading deletion" finalizer, which will be added to the finalizers list of the owning object, and removed by the GC when all dependents are deleted. The user can watch for the deletion event of the owning object to ensure the cascading deletion process has completed.


---
***THE REST IS FOR ARCHIVAL PURPOSES***
---

# Considered and Rejected Designs

# 1. Tombstone + GC

## Reasons of rejection

* It likely would conflict with our plan in the future to use all resources as their own tombstones, once the registry supports multi-object transaction.
* The TTL of the tombstone is hand-waving, there is no guarantee that the value of the TTL is long enough.
* This design is essentially the same as the selected design, with the tombstone as an extra element. The benefit the extra complexity buys is that a parent object can be deleted immediately even if the user wants to orphan the children. The benefit doesn't justify the complexity.


## API Changes

```
type DeleteOptions struct {
	…
	OrphanChildren bool
}
```

**DeleteOptions.OrphanChildren**: allows a user to express whether the child objects should be orphaned.

```
type ObjectMeta struct {
	...	
	ParentReferences []ObjectReference
}
```

**ObjectMeta.ParentReferences**: links the resource to the parent resources. For example, a replica set `R` created by a deployment `D` should have an entry in ObjectMeta.ParentReferences pointing to `D`. The link should be set when the child object is created. It can be updated after the creation.

```
type Tombstone struct {
    unversioned.TypeMeta
    ObjectMeta
	UID types.UID
}
```

**Tombstone**: a tombstone is created when an object is deleted and the user requires the children to be orphaned.
**Tombstone.UID**: the UID of the original object.

## New components

The only new component is the Garbage Collector, which consists of a scanner, a garbage processor, and a propagator.
* Scanner:
  * Uses the discovery API to detect all the resources supported by the system.
    * For performance reasons, resources can be marked as not participating cascading deletion in the discovery info, then the GC will not monitor them.
  * Periodically scans all resources in the system and adds each object to the *Dirty Queue*.

* Garbage Processor:
  * Consists of the *Dirty Queue* and workers.
  * Each worker:
    * Dequeues an item from *Dirty Queue*.
    * If the item's ParentReferences is empty, continues to process the next item in the *Dirty Queue*.
    * Otherwise checks each entry in the ParentReferences:
      * If a parent exists, continues to check the next parent.
      * If a parent doesn't exist, checks if a tombstone standing for the parent exists.
    * If the step above shows no parent nor tombstone exists, requests the API server to delete the item. That is, only if ***all*** parents are non-existent, and none of them have tombstones, the child object will be garbage collected.
    * Otherwise removes the item's ParentReferences to non-existent parents.

* Propagator:
  * The Propagator is for optimization, not for correctness.
  * Maintains a DAG of parent-child relations. This DAG stores only name/uid/orphan triplets, not the entire body of every item.
  * Consists of an *Event Queue* and a single worker.
  * Watches for create/update/delete events for all resources that participating cascading deletion, enqueues the events to the *Event Queue*.
  * Worker:
    * Dequeues an item from the *Event Queue*.
    * If the item is an creation or update, then updates the DAG accordingly.
      * If the object has a parent and the parent doesn’t exist in the DAG yet, then apart from adding the object to the DAG, also enqueues the object to the *Dirty Queue*.
    * If the item is a deletion, then removes the object from the DAG, and enqueues all its children to the *Dirty Queue*.
  * The propagator shouldn't need to do any RPCs, so a single worker should be sufficient. This makes locking easier.
  * With the Propagator, we *only* need to run the Scanner when starting the Propagator to populate the DAG and the *Dirty Queue*.

## Changes to existing components

* Storage: we should add a REST storage for Tombstones. The index should be UID rather than namespace/name.

* API Server: when handling a deletion request, if DeleteOptions.OrphanChildren is true, then the API Server either creates a tombstone with TTL if the tombstone doesn't exist yet, or updates the TTL of the existing tombstone. The API Server deletes the object after the tombstone is created.

* Controllers: when creating child objects, controllers need to fill up their ObjectMeta.ParentReferences field. Objects that don’t have a parent should have the namespace object as the parent.

## Comparison with the selected design

The main difference between the two designs is when to update the ParentReferences. In design #1, because a tombstone is created to indicate "orphaning" is desired, the updates to ParentReferences can be deferred until the deletion of the tombstone. In design #2, the updates need to be done before the parent object is deleted from the registry.

* Advantages of "Tombstone + GC" design
  * Faster to free the resource name compared to using finalizers. The original object can be deleted to free the resource name once the tombstone is created, rather than waiting for the finalizers to update all children’s ObjectMeta.ParentReferences.
* Advantages of "Finalizer Framework + GC"
  * The finalizer framework is needed for other purposes as well.


# 2. Recovering from abnormal cascading deletion

## Reasons of rejection

* Not a goal
* Tons of work, not feasible in the near future

In case the garbage collector is mistakenly deleting objects, we should provide mechanism to stop the garbage collector and restore the objects.

* Stopping the garbage collector

  We will add a "--enable-garbage-collector" flag to the controller manager binary to indicate if the garbage collector should be enabled. Admin can stop the garbage collector in a running cluster by restarting the kube-controller-manager with --enable-garbage-collector=false.

* Restoring mistakenly deleted objects
  * Guidelines
    * The restoration should be implemented as a roll-forward rather than a roll-back, because likely the state of the cluster (e.g., available resources on a node) has changed since the object was deleted.
    * Need to archive the complete specs of the deleted objects.
    * The content of the archive is sensitive, so the access to the archive subjects to the same authorization policy enforced on the original resource.
    * States should be stored in etcd. All components should remain stateless.

  * A preliminary design

    This is a generic design for “undoing a deletion”, not specific to undoing cascading deletion.
    * Add a `/archive` sub-resource to every resource, it's used to store the spec of the deleted objects.
    * Before an object is deleted from the registry, the API server clears fields like DeletionTimestamp, then creates the object in /archive and sets a TTL.
    * Add a `kubectl restore` command, which takes a resource/name pair as input, creates the object with the spec stored in the /archive, and deletes the archived object.




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/garbage-collection.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
