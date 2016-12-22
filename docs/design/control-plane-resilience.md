# Kubernetes and Cluster Federation Control Plane Resilience

## Long Term Design and Current Status

### by Quinton Hoole, Mike Danese and Justin Santa-Barbara

### December 14, 2015

## Summary

Some amount of confusion exists around how we currently, and in future
want to ensure resilience of the Kubernetes (and by implication
Kubernetes Cluster Federation) control plane.  This document is an attempt to capture that
definitively. It covers areas including self-healing, high
availability, bootstrapping and recovery.  Most of the information in
this document already exists in the form of github comments,
PR's/proposals, scattered documents, and corridor conversations, so
document is primarily a consolidation and clarification of existing
ideas.

## Terms

* **Self-healing:** automatically restarting or replacing failed
  processes and machines without human intervention
* **High availability:** continuing to be available and work correctly
  even if some components are down or uncontactable.  This typically
  involves multiple replicas of critical services, and a reliable way
  to find available replicas. Note that it's possible (but not
  desirable) to have high
  availability properties (e.g. multiple replicas) in the absence of
  self-healing properties (e.g. if a replica fails, nothing replaces
  it). Fairly obviously, given enough time, such systems typically
  become unavailable (after enough replicas have failed).
* **Bootstrapping**: creating an empty cluster from nothing
* **Recovery**: recreating a non-empty cluster after perhaps
  catastrophic failure/unavailability/data corruption

## Overall Goals

1. **Resilience to single failures:** Kubernetes clusters constrained
   to single availability zones should be resilient to individual
   machine and process failures by being both self-healing and highly
   available (within the context of such individual failures).
1. **Ubiquitous resilience by default:** The default cluster creation
   scripts for (at least) GCE, AWS and basic bare metal should adhere
   to the above (self-healing and high availability) by default (with
   options available to disable these features to reduce control plane
   resource requirements if so required).  It is hoped that other
   cloud providers will also follow the above guidelines, but the
   above 3 are the primary canonical use cases.
1. **Resilience to some correlated failures:** Kubernetes clusters
   which span multiple availability zones in a region should by
   default be resilient to complete failure of one entire availability
   zone (by similarly providing self-healing and high availability in
   the default cluster creation scripts as above).
1. **Default implementation shared across cloud providers:** The
   differences between the default implementations of the above for
   GCE, AWS and basic bare metal should be minimized.  This implies
   using shared libraries across these providers in the default
   scripts in preference to highly customized implementations per
   cloud provider.  This is not to say that highly differentiated,
   customized per-cloud cluster creation processes (e.g. for GKE on
   GCE, or some hosted Kubernetes provider on AWS) are discouraged.
   But those fall squarely outside the basic cross-platform OSS
   Kubernetes distro.
1. **Self-hosting:** Where possible, Kubernetes's existing mechanisms
   for achieving system resilience (replication controllers, health
   checking, service load balancing etc) should be used in preference
   to building a separate set of mechanisms to achieve the same thing.
   This implies that self hosting (the kubernetes control plane on
   kubernetes) is strongly preferred, with the caveat below.
1. **Recovery from catastrophic failure:** The ability to quickly and
   reliably recover a cluster from catastrophic failure is critical,
   and should not be compromised by the above goal to self-host
   (i.e. it goes without saying that the cluster should be quickly and
   reliably recoverable, even if the cluster control plane is
   broken). This implies that such catastrophic failure scenarios
   should be carefully thought out, and the subject of regular
   continuous integration testing, and disaster recovery exercises.

## Relative Priorities

1. **(Possibly manual) recovery from catastrophic failures:** having a
Kubernetes cluster, and all applications running inside it, disappear forever
perhaps is the worst possible failure mode. So it is critical that we be able to
recover the applications running inside a cluster from such failures in some
well-bounded time period.
    1. In theory a cluster can be recovered by replaying all API calls
       that have ever been executed against it, in order, but most
       often that state has been lost, and/or is scattered across
       multiple client applications or groups. So in general it is
       probably infeasible.
    1. In theory a cluster can also be recovered to some relatively
       recent non-corrupt backup/snapshot of the disk(s) backing the
       etcd cluster state. But we have no default consistent
       backup/snapshot, verification or restoration process.  And we
       don't routinely test restoration, so even if we did routinely
       perform and verify backups, we have no hard evidence that we
       can in practise effectively recover from catastrophic cluster
       failure or data corruption by restoring from these backups. So
       there's more work to be done here.
1. **Self-healing:** Most major cloud providers provide the ability to
   easily and automatically replace failed virtual machines within a
   small number of minutes (e.g. GCE
   [Auto-restart](https://cloud.google.com/compute/docs/instances/setting-instance-scheduling-options#autorestart)
   and Managed Instance Groups,
   AWS[ Auto-recovery](https://aws.amazon.com/blogs/aws/new-auto-recovery-for-amazon-ec2/)
   and [Auto scaling](https://aws.amazon.com/autoscaling/) etc). This
   can fairly trivially be used to reduce control-plane down-time due
   to machine failure to a small number of minutes per failure
   (i.e. typically around "3 nines" availability), provided that:
    1. cluster persistent state (i.e. etcd disks) is either:
        1. truely persistent (i.e. remote persistent disks), or
        1. reconstructible (e.g. using etcd [dynamic member
           addition](https://github.com/coreos/etcd/blob/master/Documentation/runtime-configuration.md#add-a-new-member)
           or [backup and
           recovery](https://github.com/coreos/etcd/blob/master/Documentation/admin_guide.md#disaster-recovery)).
    1. and boot disks are either:
        1. truely persistent (i.e. remote persistent disks), or
        1. reconstructible (e.g. using boot-from-snapshot,
           boot-from-pre-configured-image or
           boot-from-auto-initializing image).
1. **High Availability:** This has the potential to increase
   availability above the approximately "3 nines" level provided by
   automated self-healing, but it's somewhat more complex, and
   requires additional resources (e.g. redundant API servers and etcd
   quorum members).  In environments where cloud-assisted automatic
   self-healing might be infeasible (e.g. on-premise bare-metal
   deployments), it also gives cluster administrators more time to
   respond (e.g. replace/repair failed machines) without incurring
   system downtime.

## Design and Status (as of December 2015)

<table>
<tr>
<td><b>Control Plane Component</b></td>
<td><b>Resilience Plan</b></td>
<td><b>Current Status</b></td>
</tr>
<tr>
<td><b>API Server</b></td>
<td>

Multiple stateless, self-hosted, self-healing API servers behind a HA
load balancer, built out by the default "kube-up" automation on GCE,
AWS and basic bare metal (BBM). Note that the single-host approach of
having etcd listen only on localhost to ensure that only API server can
connect to it will no longer work, so alternative security will be
needed in the regard (either using firewall rules, SSL certs, or
something else). All necessary flags are currently supported to enable
SSL between API server and etcd (OpenShift runs like this out of the
box), but this needs to be woven into the "kube-up" and related
scripts.  Detailed design of self-hosting and related bootstrapping
and catastrophic failure recovery will be detailed in a separate
design doc.

</td>
<td>

No scripted self-healing or HA on GCE, AWS or basic bare metal
currently exists in the OSS distro. To be clear, "no self healing"
means that even if multiple e.g. API servers are provisioned for HA
purposes, if they fail, nothing replaces them, so eventually the
system will fail. Self-healing and HA can be set up
manually by following documented instructions, but this is not
currently an automated process, and it is not tested as part of
continuous integration. So it's probably safest to assume that it
doesn't actually work in practise.

</td>
</tr>
<tr>
<td><b>Controller manager and scheduler</b></td>
<td>

Multiple self-hosted, self healing warm standby stateless controller
managers and schedulers with leader election and automatic failover of API
server clients, automatically installed by default "kube-up" automation.

</td>
<td>As above.</td>
</tr>
<tr>
<td><b>etcd</b></td>
<td>

Multiple (3-5) etcd quorum members behind a load balancer with session
affinity (to prevent clients from being bounced from one to another).

Regarding self-healing, if a node running etcd goes down, it is always necessary
to do three things:
<ol>
<li>allocate a new node (not necessary if running etcd as a pod, in
which case specific measures are required to prevent user pods from
interfering with system pods, for example using node selectors as
described in <A HREF="),
<li>start an etcd replica on that new node, and
<li>have the new replica recover the etcd state.
</ol>
In the case of local disk (which fails in concert with the machine), the etcd
state must be recovered from the other replicas. This is called
<A HREF="https://github.com/coreos/etcd/blob/master/Documentation/runtime-configuration.md#add-a-new-member">
dynamic member addition</A>.

In the case of remote persistent disk, the etcd state can be recovered by
attaching the remote persistent disk to the replacement node, thus the state is
recoverable even if all other replicas are down.

There are also significant performance differences between local disks and remote
persistent disks. For example, the
<A HREF="https://cloud.google.com/compute/docs/disks/#comparison_of_disk_types">
sustained throughput local disks in GCE is approximatley 20x that of remote
disks</A>.

Hence we suggest that self-healing be provided by remotely mounted persistent
disks in non-performance critical, single-zone cloud deployments. For
performance critical installations, faster local SSD's should be used, in which
case remounting on node failure is not an option, so
<A HREF="https://github.com/coreos/etcd/blob/master/Documentation/runtime-configuration.md ">
etcd runtime configuration</A> should be used to replace the failed machine.
Similarly, for cross-zone self-healing, cloud persistent disks are zonal, so
automatic <A HREF="https://github.com/coreos/etcd/blob/master/Documentation/runtime-configuration.md">
runtime configuration</A> is required. Similarly, basic bare metal deployments
cannot generally rely on remote persistent disks, so the same approach applies
there.
</td>
<td>
<A HREF="http://kubernetes.io/v1.1/docs/admin/high-availability.html">
Somewhat vague instructions exist</A> on how to set some of this up manually in
a self-hosted configuration. But automatic bootstrapping and self-healing is not
described (and is not implemented for the non-PD cases). This all still needs to
be automated and continuously tested.
</td>
</tr>
</table>


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/control-plane-resilience.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
