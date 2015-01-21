# Availability

This document collects advice on reasoning about and provisioning for high-availability when using Kubernetes clusters.

## Failure modes

This is an incomplete list of things that could go wrong, and how to deal with it.

Root causes:
  - VM(s) shutdown
  - network partition within cluster, or between cluster and users.
  - crashes in Kubernetes software 
  - data loss or unavailability from storage
  - operator error misconfigures kubernetes software or application software.

Specific scenarios:
  - Apiserver VM shutdown or apiserver crashing
    - Results
      - unable to stop, update, or start new pods, services, replication controller
      - existing pods and services should continue to work normally, unless they depend on the Kubernetes API
    - Mitigations
      - Use cloud provider best practices for improving availability of a VM, such as automatic restart and reliable
        storage for writeable state (GCE PD or AWS EBS volume).
      - High-availability (replicated) APIserver is a planned feature for Kubernetes.  Will tolerate one or more
        similtaneous apiserver failures.
      - Multiple independent clusters will tolerate failure of all apiservers in one cluster.  
  - Apiserver backing storage lost
    - Results
      - apiserver should fail to come up.
      - kubelets will not be able to reach it but will continute to run the same pods and provide the same service proxying.
      - manual recovery or recreation of apiserver state necessary before apiserver is restarted.
    - Mitigations
      - High-availability (replicated) APIserver is a planned feature for Kubernetes.  Each apiserver has independent
        storage.  Etcd will recover from loss of one member.  Risk of total data loss greatly reduced.
      - snapshot PD/EBS-volume periodically
  - Supporting services (node controller, replication controller manager, scheduler, etc) VM shutdown or crashes
    - currently those are colocated with the apiserver, and their unavailability has similar consequences as apiserver
    - in future, these will be replicated as well and may not be co-located
    - they do not have own persistent state
  - Node (thing that runs kubelet and kube-proxy and pods) shutdown
    - Results
      - pods on that Node stop running
    - Mitigations
      - replication controller should be used to restart copy of the pod elsewhere
      - service should be used to hide changes in the pod IP address after restart
      - applications (containers) should tolerate unexpected restarts
  - Kubelet software fault
    - Results
      - crashing kubelet cannot start new pods on the node
      - kubelet might delete the pods or not
      - node marked unhealthy
      - replication controllers start new pods elsewhere
    - Mitigations
      - same as for Node shutdown case
  - Cluster operator error
    - Results:
      - loss of pods, services, etc
      - lost of apiserver backing store
      - users unable to read API
      - etc
    - Mitigations
      - run additional cluster(s) and do not make changes to all at once.
      - snapshot apiserver PD/EBS-volume periodically

## Chosing Multiple Kubernetes Clusters

You may want to set up multiple kubernetes clusters, both to
 to have clusters in different regions to be nearer to your users; and to tolerate failures and/or invasive maintenance.

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
  - reduced operational overhead, though advanatage diminished as ops tooling and processes matures.
  - reduced costs for per-cluster CPU, Memory, and Disk needs (apiserver etc...); though small as a percentage
    of overall cluster cost for medium to large clusters.
Reasons you might want multiple clusters:
  - strict security policies requiring isolation of one class of work from another (but, see Partitioning Clusters
    below).
  - test clusters to canary new Kubernetes releases or other cluster software.

### Selecting the right number of clusters
The selection of the number of kubernetes clusters may be a relatively static choice, only revisted occasionally.
By contrast, the number of nodes in a cluster and the number of pods in a service may be change frequently according to
load and growth.

To pick the number of clusters, first, decide which regions you need to be in to have adequete latency to all your end users, for services that will run
on Kubernetes (if you use a Content Distribution Network, the latency requirements for the CDN-hosted content need not
be considered).  For example, a company with a global customer base might decide to have clusters in US, EU, AP, and SA regions.   That is the minimum number of
Kubernetes clusters.  Call this `R`

Second, decide how many clusters should be able to be unavailable at the same time, in order to meet your availability
goals.  If you are not sure, then 1 is a good number.  Call this `U`.   Reasons for unavailability include:
 - IaaS provider unavailable
 - cluster operator error
 - Kubernetes software fault

If you are able and willing to fail over to a different region than some customers in the event of a cluster failure,
then you need R + U clusters.  If you want to ensure low latency for all users in the event of a cluster failure, you
need to have R*U clusters (U in each of R regions).  In either case, put each cluster in a different zone.

Finally, if any of your clusters would need to be larger than the maximum number of nodes for a Kubernetes cluster, then
you may need even more clusters.  Our roadmap (
https://github.com/GoogleCloudPlatform/kubernetes/blob/24e59de06e4da61f5dafd4cd84c9340a2c0d112f/docs/roadmap.md)
calls for maximum 100 node clusters at v1.0 and maximum 1000 node clusters in the middle of 2015.

## Working with multiple clusters

When you have multiple clusters, you would typically copies of a given service in each cluster and put each of those
service instances behind a load balancer (AWS Elastic Load Balancer, GCE Forwarding Rule or HTTP Load Balancer), so that
failures of a single cluster are not visible to end users.

