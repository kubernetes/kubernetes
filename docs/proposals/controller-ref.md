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

# ControllerRef proposal

Author: gmarek@
Last edit: 2016-05-11
Status: raw

Approvers:
- [ ] briangrant
- [ ] dbsmith

**Table of Contents**

- [Goal of ControllerReference](#goal-of-setreference)
- [Non goals](#non-goals)
- [API and semantic changes](#api-and-semantic-changes)
- [Upgrade/downgrade procedure](#upgradedowngrade-procedure)
- [Orphaning/adoption](#orphaningadoption)
- [Implementation plan (sketch)](#implementation-plan-sketch)
- [Considered alternatives](#considered-alternatives)

# Goal of ControllerReference

Main goal of `ControllerReference` effort is to solve a problem of overlapping controllers that fight over some resources (e.g. `ReplicaSets` fighting with `ReplicationControllers` over `Pods`), which cause serious [problems](https://github.com/kubernetes/kubernetes/issues/24433) such as exploding memory of Controller Manager.

We don’t want to have (just) an in-memory solution, as we don’t want a Controller Manager crash to cause massive changes in object ownership in the system. I.e. we need to persist the information about "owning controller".

Secondary goal of this effort is to improve performance of various controllers and schedulers, by removing the need for expensive lookup for all matching "controllers".

# Non goals

Cascading deletion is not a goal of this effort. Cascading deletion will use `ownerReferences`, which is a [separate effort](garbage-collection.md).

`ControllerRef` will extend `OwnerReference` and reuse machinery written for it (GarbageCollector, adoption/orphaning logic).

# API and semantic changes

There will be a new API field in the `OwnerReference` in which we will store an information if given owner is a managing controller:

```
OwnerReference {
    …
    Controller bool
    …
}
```

From now on by `ControllerRef` we mean an `OwnerReference` with `Controller=true`.

Most controllers (all that manage collections of things defined by label selector) will have slightly changed semantics: currently controller owns an object if its selector matches object’s labels and if it doesn't notice an older controller of the same kind that also matches the object's labels, but after introduction of `ControllerReference` a controller will own an object iff selector matches labels and the `OwnerReference` with `Controller=true`points to it.

If the owner's selector or owned object's labels change, the owning controller will be responsible for orphaning (clearing `Controller` field in the `OwnerReference` and/or deleting `OwnerReference` altogether) objects, after which adoption procedure (setting `Controller` field in one of `OwnerReferencec` and/or adding new `OwnerReferences`) might occur, if another controller has a selector matching.

For debugging purposes we want to add an `adoptionTime` annotation prefixed with `kubernetes.io/` which will keep the time of last controller ownership transfer.

# Upgrade/downgrade procedure

Because `ControllerRef` will be a part of `OwnerReference` effort it will have the same upgrade/downgrade procedures.

# Orphaning/adoption

Because `ControllerRef` will be a part of `OwnerReference` effort it will have the same orphaning/adoption procedures.

Controllers will orphan objects they own in two cases:
* Change of label/selector causing selector to stop matching labels (executed by the controller)
* Deletion of a controller with `Orphaning=true` (executed by the GarbageCollector)

We will need a secondary orphaning mechanism in case of unclean controller deletion:
* GarbageCollector will remove `ControllerRef` from objects that no longer points to existing controllers

Controller will adopt (set `Controller` field in the `OwnerReference` that points to it) an object whose labels match its selector iff:
* there are no `OwnerReferences` with `Controller` set to true in `OwnerReferences` array
* `DeletionTimestamp` is not set
and
* Controller is the first controller that will manage to adopt the Pod from all Controllers that have matching label selector and don't have `DeletionTimestamp` set.

By design there are possible races during adoption if multiple controllers can own a given object.

To prevent re-adoption of an object during deletion the `DeletionTimestamp` will be set when deletion is starting. When a controller has a non-nil `DeletionTimestamp` it won’t take any actions except updating its `Status` (in particular it won’t adopt any objects).

# Implementation plan (sketch):

* Add API field for `Controller`,
* Extend `OwnerReference` adoption procedure to set a `Controller` field in one of the owners,
* Update all affected controllers to respect `ControllerRef`.

Necessary related work:
* `OwnerReferences` are correctly added/deleted,
* GarbageCollector removes dangling references,
* Controllers don't take any meaningfull actions when `DeletionTimestamps` is set.

# Considered alternatives

* Generic "ReferenceController": centralized component that managed adoption/orphaning
    * Dropped because: hard to write something that will work for all imaginable 3rd party objects, adding hooks to framework makes it possible for users to write their own logic
* Separate API field for `ControllerRef` in the ObjectMeta.
    * Dropped because: nontrivial relationship between `ControllerRef` and `OwnerReferences` when it comes to deletion/adoption.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/controller-ref.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
