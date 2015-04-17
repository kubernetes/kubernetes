# High Availability of Daemons in Kubernetes
This document serves as a proposal for high availability of the master daemons in kubernetes.

## Design Options
1. Hot Standby Daemons: In this scenario, data and state are shared between the two deamons such that an immediate failure in one daemon causes the the standby deamon to take over exactly where the failed daemon had left off.  This would be an ideal solution for kubernetes, however it poses a series of challenges in the case of controllers where daemon-state is cached locally and not persisted in a transactional way to a storage facility.  As a result, we are **NOT** planning on this approach. 

2. **Cold Standby Daemons**: In this scenario there is only one active daemon acting as the master and additional daemons in a standby mode.  Data and state are not shared between the active and standby daemons, so when a failure occurs the standby daemon that becomes the master must determine the current state of the system before resuming functionality.

3. Stateless load-balanced Daemons: Stateless daemons, such as the apiserver, can simply load-balance across any number of servers that are currently running.  Their general availability can be continuously updated, or published, such that load balancing only occurs across active participants.  This aspect of HA is outside of the scope of *this* proposal because there is already a partial implementation in the apiserver.


## Design Discussion Notes on Leader Election
For a very simple example of proposed behavior see: 
* https://github.com/rrati/etcd-ha
* go get github.com/rrati/etcd-ha

In HA, the apiserver will be a gateway to etcd. It will provide an api for becoming master, updating the master lease, and releasing the lease.  This api is daemon agnostic, so to become the master the client will need to provide the daemon type and the lease duration when attemping to become master.  The apiserver will attempt to create a key in etcd based on the daemon type that contains the client's hostname/ip and port information. This key will be created with a ttl from the lease duration provided in the request.  Failure to create this key means there is already a master of that daemon type, and the error from etcd will propigate to the client.  Successfully creating the key means the client making the request is the master.  When updating the lease, the apiserver will update the existing key with a new ttl.  The location in etcd for the HA keys is TBD.

Leader election is first come, first serve.  The first daemon of a specific type to request leadership will become the master.  All other daemons of that type will fail until the current leader releases the lease or fails to update the lease within the expiration time.  On startup, all daemons should attempt to become master.  The daemon that succeeds is the master and should perform all functions of that daemon.  The daemons that fail to become the master should not perform any tasks and sleep for their lease duration and then attempt to become the master again.

The daemon that becomes master should create a Go routine to manage the lease.  This process should be created with a channel that the main daemon process can use to release the master lease.  Otherwise, this process will update the lease and sleep, waiting for the next update time or notification to release the lease.  If there is a failure to update the lease, this process should force the entire daemon to exit.  Daemon exit is meant to prevent potential split-brain conditions.  Daemon restart is implied in this scenario, by either the init system (systemd), or possible watchdog processes.  (See Design Discussion Notes)

## Options added to daemons with HA functionality
Some command line options would be added to daemons that can do HA:

* Lease Duration - How long a daemon can be master

* Number of Missed Lease Updates - How many updates can be missed before the lease as the master is lost

## Design Discussion Notes on Scheduler/Controller
Some daemons, such as the controller-manager, may fork numerous go routines to perform tasks in parallel.  Trying to keep track of all these processes and shut them down cleanly is untenable.  If a master daemon loses leadership then the whole daemon should exit with an exit code indicating that the daemon is not the master.  The daemon should be restarted by a monitoring system, such as systemd, or a software watchdog.

## Open Questions:
* Is there a desire to keep track of all nodes for a specific daemon type?
