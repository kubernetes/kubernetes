<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Design Principles

Principles to follow when extending Kubernetes.

## API

See also the [API conventions](../devel/api-conventions.md).

* All APIs should be declarative.
* API objects should be complementary and composable, not opaque wrappers.
* The control plane should be transparent -- there are no hidden internal APIs.
* The cost of API operations should be proportional to the number of objects intentionally operated upon. Therefore, common filtered lookups must be indexed. Beware of patterns of multiple API calls that would incur quadratic behavior.
* Object status must be 100% reconstructable by observation. Any history kept must be just an optimization and not required for correct operation.
* Cluster-wide invariants are difficult to enforce correctly. Try not to add them. If you must have them, don't enforce them atomically in master components, that is contention-prone and doesn't provide a recovery path in the case of a bug allowing the invariant to be violated. Instead, provide a series of checks to reduce the probability of a violation, and make every component involved able to recover from an invariant violation.
* Low-level APIs should be designed for control by higher-level systems. Higher-level APIs should be intent-oriented (think SLOs) rather than implementation-oriented (think control knobs).

## Control logic

* Functionality must be *level-based*, meaning the system must operate correctly given the desired state and the current/observed state, regardless of how many intermediate state updates may have been missed. Edge-triggered behavior must be just an optimization.
* Assume an open world: continually verify assumptions and gracefully adapt to external events and/or actors. Example: we allow users to kill pods under control of a replication controller; it just replaces them.
* Do not define comprehensive state machines for objects with behaviors associated with state transitions and/or "assumed" states that cannot be ascertained by observation.
* Don't assume a component's decisions will not be overridden or rejected, nor for the component to always understand why. For example, etcd may reject writes. Kubelet may reject pods. The scheduler may not be able to schedule pods. Retry, but back off and/or make alternative decisions.
* Components should be self-healing. For example, if you must keep some state (e.g., cache) the content needs to be periodically refreshed, so that if an item does get erroneously stored or a deletion event is missed etc, it will be soon fixed, ideally on timescales that are shorter than what will attract attention from humans.
* Component behavior should degrade gracefully. Prioritize actions so that the most important activities can continue to function even when overloaded and/or in states of partial failure.

## Architecture

* Only the apiserver should communicate with etcd/store, and not other components (scheduler, kubelet, etc.).
* Compromising a single node shouldn't compromise the cluster.
* Components should continue to do what they were last told in the absence of new instructions (e.g., due to network partition or component outage).
* All components should keep all relevant state in memory all the time. The apiserver should write through to etcd/store, other components should write through to the apiserver, and they should watch for updates made by other clients.
* Watch is preferred over polling.

## Extensibility

TODO: pluggability

## Bootstrapping

* [Self-hosting](http://issue.k8s.io/246) of all components is a goal.
* Minimize the number of dependencies, particularly those required for steady-state operation.
* Stratify the dependencies that remain via principled layering.
* Break any circular dependencies by converting hard dependencies to soft dependencies.
  * Also accept that data from other components from another source, such as local files, which can then be manually populated at bootstrap time and then continuously updated once those other components are available.
  * State should be rediscoverable and/or reconstructable.
  * Make it easy to run temporary, bootstrap instances of all components in order to create the runtime state needed to run the components in the steady state; use a lock (master election for distributed components, file lock for local components like Kubelet) to coordinate handoff. We call this technique "pivoting".
  * Have a solution to restart dead components. For distributed components, replication works well. For local components such as Kubelet, a process manager or even a simple shell loop works.

## Availability

TODO

## General principles

* [Eric Raymond's 17 UNIX rules](https://en.wikipedia.org/wiki/Unix_philosophy#Eric_Raymond.E2.80.99s_17_Unix_Rules)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/principles.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
