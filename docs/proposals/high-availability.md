<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# High Availability of Scheduling and Controller Components in Kubernetes

This document serves as a proposal for high availability of the scheduler and controller components in Kubernetes.  This proposal is intended to provide a simple High Availability api for Kubernetes components with the potential to extend to services running on Kubernetes.  Those services would be subject to their own constraints.

## Design Options

For complete reference see [this](https://www.ibm.com/developerworks/community/blogs/RohitShetty/entry/high_availability_cold_warm_hot?lang=en)

1. Hot Standby: In this scenario, data and state are shared between the two components such that an immediate failure in one component causes the standby daemon to take over exactly where the failed component had left off.  This would be an ideal solution for Kubernetes, however it poses a series of challenges in the case of controllers where component-state is cached locally and not persisted in a transactional way to a storage facility.  This would also introduce additional load on the apiserver, which is not desirable.  As a result, we are **NOT** planning on this approach at this time.

2. **Warm Standby**: In this scenario there is only one active component acting as the master and additional components running but not providing service or responding to requests.  Data and state are not shared between the active and standby components.  When a failure occurs, the standby component that becomes the master must determine the current state of the system before resuming functionality.  This is the approach that this proposal will leverage.

3. Active-Active (Load Balanced): Clients can simply load-balance across any number of servers that are currently running.  Their general availability can be continuously updated, or published, such that load balancing only occurs across active participants.  This aspect of HA is outside of the scope of *this* proposal because there is already a partial implementation in the apiserver.

## Design Discussion Notes on Leader Election

Implementation References:
* [zookeeper](http://zookeeper.apache.org/doc/trunk/recipes.html#sc_leaderElection)
* [etcd](https://groups.google.com/forum/#!topic/etcd-dev/EbAa4fjypb4)
* [initialPOC](https://github.com/rrati/etcd-ha)

In HA, the apiserver will provide an api for sets of replicated clients to do master election: acquire the lease, renew the lease, and release the lease.  This api is component agnostic, so a client will need to provide the component type and the lease duration when attempting to become master.  The lease duration should be tuned per component.  The apiserver will attempt to create a key in etcd based on the component type that contains the client's hostname/ip and port information. This key will be created with a ttl from the lease duration provided in the request.  Failure to create this key means there is already a master of that component type, and the error from etcd will propagate to the client.  Successfully creating the key means the client making the request is the master.  Only the current master can renew the lease.  When renewing the lease, the apiserver will update the existing key with a new ttl.  The location in etcd for the HA keys is TBD.

The first component to request leadership will become the master.  All other components of that type will fail until the current leader releases the lease, or fails to renew the lease within the expiration time.  On startup, all components should attempt to become master.  The component that succeeds becomes the master, and should perform all functions of that component.  The components that fail to become the master should not perform any tasks and sleep for their lease duration and then attempt to become the master again. A clean shutdown of the leader will cause a release of the lease and a new master will be elected.

The component that becomes master should create a thread to manage the lease.  This thread should be created with a channel that the main process can use to release the master lease.  The master should release the lease in cases of an unrecoverable error and clean shutdown.  Otherwise, this process will renew the lease and sleep, waiting for the next renewal time or notification to release the lease.  If there is a failure to renew the lease, this process should force the entire component to exit.  Daemon exit is meant to prevent potential split-brain conditions.  Daemon restart is implied in this scenario, by either the init system (systemd), or possible watchdog processes.  (See Design Discussion Notes)

## Options added to components with HA functionality

Some command line options would be added to components that can do HA:

* Lease Duration - How long a component can be master

## Design Discussion Notes

Some components may run numerous threads in order to perform tasks in parallel.  Upon losing master status, such components should exit instantly instead of attempting to gracefully shut down such threads.  This is to ensure that, in the case there's some propagation delay in informing the threads they should stop, the lame-duck threads won't interfere with the new master.  The component should exit with an exit code indicating that the component is not the master.  Since all components will be run by systemd or some other monitoring system, this will just result in a restart.

There is a short window after a new master acquires the lease, during which data from the old master might be committed.  This is because there is currently no way to condition a write on its source being the master.  Having the daemons exit shortens this window but does not eliminate it.  A proper solution for this problem will be addressed at a later date.  The proposed solution is:

1. This requires transaction support in etcd (which is already planned - see [coreos/etcd#2675](https://github.com/coreos/etcd/pull/2675))

2. The entry in etcd that is tracking the lease for a given component (the "current master" entry) would have as its value the host:port of the lease-holder (as described earlier) and a sequence number. The sequence number is incremented whenever a new master gets the lease.

3. Master replica is aware of the latest sequence number.

4. Whenever master replica sends a mutating operation to the API server, it includes the sequence number.

5. When the API server makes the corresponding write to etcd, it includes it in a transaction that does a compare-and-swap on the "current master" entry (old value == new value == host:port and sequence number from the replica that sent the mutating operation). This basically guarantees that if we elect the new master, all transactions coming from the old master will fail. You can think of this as the master attaching a "precondition" of its belief about who is the latest master.

## Open Questions

* Is there a desire to keep track of all nodes for a specific component type?




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/high-availability.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
