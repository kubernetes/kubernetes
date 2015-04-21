# Availability

This document collects advice on reasoning about and provisioning for high-availability when using Kubernetes clusters.

## Failure modes

This is an incomplete list of things that could go wrong, and how to deal with them.

Root causes:
  - VM(s) shutdown
  - network partition within cluster, or between cluster and users.
  - crashes in Kubernetes software 
  - data loss or unavailability of persistent storage (e.g. GCE PD or AWS EBS volume).
  - operator error misconfigures kubernetes software or application software.

Specific scenarios:
  - Apiserver VM shutdown or apiserver crashing
    - Results
      - unable to stop, update, or start new pods, services, replication controller
      - existing pods and services should continue to work normally, unless they depend on the Kubernetes API
  - Apiserver backing storage lost
    - Results
      - apiserver should fail to come up.
      - kubelets will not be able to reach it but will continute to run the same pods and provide the same service proxying.
      - manual recovery or recreation of apiserver state necessary before apiserver is restarted.
  - Supporting services (node controller, replication controller manager, scheduler, etc) VM shutdown or crashes
    - currently those are colocated with the apiserver, and their unavailability has similar consequences as apiserver
    - in future, these will be replicated as well and may not be co-located
    - they do not have own persistent state
  - Node (thing that runs kubelet and kube-proxy and pods) shutdown
    - Results
      - pods on that Node stop running
  - Kubelet software fault
    - Results
      - crashing kubelet cannot start new pods on the node
      - kubelet might delete the pods or not
      - node marked unhealthy
      - replication controllers start new pods elsewhere
  - Cluster operator error
    - Results:
      - loss of pods, services, etc
      - lost of apiserver backing store
      - users unable to read API
      - etc

Mitigations:
- Action: Use IaaS providers automatic VM restarting feature for IaaS VMs.
  - Mitigates: Apiserver VM shutdown or apiserver crashing
  - Mitigates: Supporting services VM shutdown or crashes

- Action use IaaS providers reliable storage (e.g GCE PD or AWS EBS volume) for VMs with apiserver+etcd.
  - Mitigates: Apiserver backing storage lost

- Action: Use Replicated APIserver feature (when complete: feature is planned but not implemented)
  - Mitigates: Apiserver VM shutdown or apiserver crashing
    - Will tolerate one or more simultaneous apiserver failures.
  - Mitigates: Apiserver backing storage lost
    - Each apiserver has independent storage.  Etcd will recover from loss of one member.  Risk of total data loss greatly reduced.

- Action: Snapshot apiserver PDs/EBS-volumes periodically
  - Mitigates: Apiserver backing storage lost
  - Mitigates: Some cases of operator error
  - Mitigates: Some cases of kubernetes software fault

- Action: use replication controller and services in front of pods
  - Mitigates: Node shutdown
  - Mitigates: Kubelet software fault

- Action: applications (containers) designed to tolerate unexpected restarts
  - Mitigates: Node shutdown
  - Mitigates: Kubelet software fault

- Action: Multiple independent clusters (and avoid making risky changes to all clusters at once)
  - Mitigates: Everything listed above.

## Choosing Multiple Kubernetes Clusters

You may want to set up multiple kubernetes clusters, both to
have clusters in different regions to be nearer to your users; and to tolerate failures and/or invasive maintenance.

### Scope of a single cluster

On IaaS providers such as Google Compute Engine or Amazon Web Services, a VM exists in a
[zone](https://cloud.google.com/compute/docs/zones) or [availability
zone](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html).
We suggest that all the VMs in a Kubernetes cluster should be in the same availability zone, because:
  - compared to having a single global Kubernetes cluster, there are fewer single-points of failure
  - compared to a cluster that spans availability zones, it is easier to reason about the availability properties of a
    single-zone cluster.
  - when the Kubernetes developers are designing the system (e.g. making assumptions about latency, bandwidth, or
    correlated failures) they are assuming all the machines are in a single data center, or otherwise closely connected.

It is okay to have multiple clusters per availability zone, though on balance we think fewer is better.
Reasons to prefer fewer clusters are:
  - improved bin packing of Pods in some cases with more nodes in one cluster.
  - reduced operational overhead (though the advantage is diminished as ops tooling and processes matures).
  - reduced costs for per-cluster fixed resource costs, e.g. apiserver VMs (but small as a percentage
    of overall cluster cost for medium to large clusters).

Reasons to have multiple clusters include:
  - strict security policies requiring isolation of one class of work from another (but, see Partitioning Clusters
    below).
  - test clusters to canary new Kubernetes releases or other cluster software.

### Selecting the right number of clusters
The selection of the number of kubernetes clusters may be a relatively static choice, only revisted occasionally.
By contrast, the number of nodes in a cluster and the number of pods in a service may be change frequently according to
load and growth.

To pick the number of clusters, first, decide which regions you need to be in to have adequete latency to all your end users, for services that will run
on Kubernetes (if you use a Content Distribution Network, the latency requirements for the CDN-hosted content need not
be considered).  Legal issues might influence this as well. For example, a company with a global customer base might decide to have clusters in US, EU, AP, and SA regions. 
Call the number of regions to be in `R`.

Second, decide how many clusters should be able to be unavailable at the same time, while still being available.  Call
the number that can be unavailable `U`.  If you are not sure, then 1 is a fine choice.

If it is allowable for load-balancing to direct traffic to any region in the event of a cluster failure, then 
then you need `R + U` clusters.  If it is not (e.g you want to ensure low latency for all users in the event of a
cluster failure), then you need to have `R * U` clusters (`U` in each of `R` regions).  In any case, try to put each cluster in a different zone.

Finally, if any of your clusters would need more than the maximum recommended number of nodes for a Kubernetes cluster, then
you may need even more clusters.  Our [roadmap](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/roadmap.md)
calls for maximum 100 node clusters at v1.0 and maximum 1000 node clusters in the middle of 2015.

## Working with multiple clusters

When you have multiple clusters, you would typically create services with the same config in each cluster and put each of those
service instances behind a load balancer (AWS Elastic Load Balancer, GCE Forwarding Rule or HTTP Load Balancer), so that
failures of a single cluster are not visible to end users.
