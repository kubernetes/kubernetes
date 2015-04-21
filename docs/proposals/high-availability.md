# High Availability of Scheduling and Controller Components in Kubernetes
This document serves as a proposal for high availability of the scheduler and controller components in kubernetes.  This proposal is intended to provide a simple High Availability api for kubernertes components only.  Extensibility beyond that scope will be subject to other constraints.

## Design Options
For complete reference see [this](https://www.ibm.com/developerworks/community/blogs/RohitShetty/entry/high_availability_cold_warm_hot?lang=en)

1. Hot Standby: In this scenario, data and state are shared between the two components such that an immediate failure in one component causes the the standby deamon to take over exactly where the failed component had left off.  This would be an ideal solution for kubernetes, however it poses a series of challenges in the case of controllers where component-state is cached locally and not persisted in a transactional way to a storage facility.  This would also introduce additional load on the apiserver, which is not desireable.  As a result, we are **NOT** planning on this approach at this time. 

2. **Warm Standby**: In this scenario there is only one active component acting as the master and additional components running by not providing service or responding to requests.  Data and state are not shared between the active and standby components.  When a failure occurs, the standby component that becomes the master must determine the current state of the system before resuming functionality.

3. Active-Active (Load Balanced): Components, such as the apiserver, can simply load-balance across any number of servers that are currently running.  Their general availability can be continuously updated, or published, such that load balancing only occurs across active participants.  This aspect of HA is outside of the scope of *this* proposal because there is already a partial implementation in the apiserver.

## Design Discussion Notes on Leader Election
Implementation References:
* [zookeeper](http://zookeeper.apache.org/doc/trunk/recipes.html#sc_leaderElection)
* [etcd](https://groups.google.com/forum/#!topic/etcd-dev/EbAa4fjypb4)
* [initialPOC](https://github.com/rrati/etcd-ha)

In HA, the apiserver will provide an api for sets of replicated clients to do master election: become master, update the lease, and release the lease.  This api is component agnostic, so a client will need to provide the component type and the lease duration when attemping to become master.  The lease duration should be tuned per component.  The apiserver will attempt to create a key in etcd based on the component type that contains the client's hostname/ip and port information. This key will be created with a ttl from the lease duration provided in the request.  Failure to create this key means there is already a master of that component type, and the error from etcd will propigate to the client.  Successfully creating the key means the client making the request is the master.  When updating the lease, the apiserver will update the existing key with a new ttl.  The location in etcd for the HA keys is TBD.

The first component to request leadership will become the master.  All other components of that type will fail until the current leader releases the lease, or fails to update the lease within the expiration time.  On startup, all components should attempt to become master.  The component that succeeds becomes the master, and should perform all functions of that component.  The components that fail to become the master should not perform any tasks and sleep for their lease duration and then attempt to become the master again. A clean shutdown of the leader will cause a release of the lease and a new master will be elected.

The component that becomes master should create a thread to manage the lease.  This thread should be created with a channel that the main process can use to release the master lease.  The master should release the lease in cases of an unrecoverable error and clean shutdown.  Otherwise, this process will update the lease and sleep, waiting for the next update time or notification to release the lease.  If there is a failure to update the lease, this process should force the entire component to exit.  Daemon exit is meant to prevent potential split-brain conditions.  Daemon restart is implied in this scenario, by either the init system (systemd), or possible watchdog processes.  (See Design Discussion Notes)

## Options added to components with HA functionality
Some command line options would be added to components that can do HA:

* Lease Duration - How long a component can be master

## Design Discussion Notes
Some components may run numerous threads in order to perform tasks in parallel. Upon losing master status, such components should exit instantly instead of attempting to gracefully shut down such threads. This is to ensure that, in the case there's some propagation delay in informing the threads they should stop, the lame-duck threads won't interfere with the new master.  The component should exit with an exit code indicating that the component is not the master.  Since all components will be run by systemd or some other monitoring system, this will just result in a restart.

## Open Questions:
* Is there a desire to keep track of all nodes for a specific component type?
