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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/job.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Ingress HA, Scheduling, and Provisioning Proposal
----


## Overview
Ingress can be used to expose a service in the kubernetes cluster:

* usually cluster admin deploys one Ingress Pod
* user creates a Ingress resource
* the Ingress Pod will list&watch All Ingress Resources in the cluster
* user out of cluster then can access service in the cluster by accessing 
  the node's ip on which Ingress Pod is running, Ingress Pod will forward 
  request into cluster based on rules defined in Ingress Resource

This just works. What's the issus then?

The issues is:

* It does not provide High Availability because client needs to know 
  the IP addresss of the node where Ingress Pod is running. In case of a 
  failure the Ingress Pod can be be moved to a different node.
* How many Ingress Pod should run in a cluster? Should all Ingress Pod 
  list&watch all Ingress Resource with out distinction? There is no way 
  to bind or schedule ingress resource to a Ingress Pod/ReplicaSet(or a 
  set of Ingress Pods/ReplicaSets), result in insufficient or excessive 
  use of resource.

## Goal
This Proposal aims to address the above issues by the following mechanism:

* Ingress HA: using keepalived and VIP to provide High Availability(mainly
  for nginx/haproxy implementation, cloud implementation usually already 
  provide HA)
* Ingress Scheduling: schedule Ingress Resource to Ingress Pod/ReplicaSet
* Ingress Provisioning: allow user to dynamically add Ingress Pod/ReplicaSet 
  on demand

## NonGoal

* Ingress ReplicaSet rolling update

## Ingress HA
(AKA: Ingress Virtual IP using keepalived)

#### High level design
* use keepalived to provide HA
* cluster admin choose a group of nodes which could be accessed out of cluster 
  and are in the same L2 broacast domain to run Ingress Pod
* deploy Ingress Pod using ReplicaSet(at least 2 replicas for HA)
* using AntiAffinity feature so that Ingress Pod created by the same Ingress 
  ReplicaSet could be scheduled to different node
* cluster admin choose a CIDR for Ingress VIP(AKA IngreeVIPCIDR)
* each Ingress Replicaset will be allocated a VIP from IngreeVIPCIDR(allocated by
  cluster admin or API server)
* Ingress Pods use host network
* Ingress Pods created by the same Ingress ReplicaSet will run keepalived, only
  one Ingress Pod will get the VIP
* users out of cluster access incluster service by the Ingress VIP 

#### Why VIP instead of round-robin DNS
A question that pops up every now and then is why we do all this stuff 
with virtual IPs rather than just use standard round-robin DNS. 
There are a few reasons:

* There is a long history of DNS libraries not respecting DNS TTLs and 
    caching the results of name lookups.
* Many apps do DNS lookups once and cache the results.
* Even if apps and libraries did proper re-resolution, the load of every 
    client re-resolving DNS over and over would be difficult to manage.

#### Challenge
* VIP is boud to Ingress ReplicaSet, how to expost it to Ingress Pod? One
  approache is using ConfigMap, but then cluster admin need allocate VIP and 
  write it to ConnfigMap, which makes automatic deployment harder. 
* All Ingress Pods created by the same Ingress ReplicaSet need know others' RIP.

## Ingress Scheduling

#### High level design

* Ingress ReplicaSets are created by cluster admin in advance
* If all Ingress Pods are saturated, it's cluster admin's duty to create 
  more Ingress ReplicaSets
* There is a Ingress Scheduler which will schedule Ingress Resources to Ingress 
  ReplicaSets
* Ingress Pod will only list&watch Ingress Resources which is scheduled on it's 
  Ingress ReplicaSet.
* Ingress Scheduler make the shceduling decision by label/selector, number of 
  Ingress Resource already bound, and some  metrics(for example, mem/cpu/bandwidth
  load of Ingress Pod)

#### Implementation
* add a IngressReplicasetName and Selector field to IngressSpec
(add as annotation during incubation)

```
type IngressSpec struct{
    /*
    ...
    */
    Selector labels.Selector
    IngressReplicasetName string
}
```

* Ingress Scheduler will bind a Ingress Resource to a Ingress ReplicaSet
* Ingress Pod will only list&watch Ingress Resources which is scheduled to 
  it's Ingress ReplicaSet
* Implement Ingress scheduler so that it could respect Selector in the first step
* In long run, we will make Ingress scheduler make scheduling decision basen on 
  some monitoring metrics(e.g. mem/cpu/bandwidth load).

#### Challenge
* Ingress Resource is bound to Ingress ReplicaSet, how to expost it to Ingress 
  Pod? 

#### TBD
* Should the scheduler bind Ingress Resource to only one Ingress ReplicaSet, or
  bind to multiple Ingress ReplicaSet?

## Ingress Provisoning

#### High level design
* Ingress ReplicaSets could be dynamically provisioned on deman, instead of 
  been created by cluster admin in advance
* If a user want a Ingress ReplicaSet to serve his Ingress Resource, he could 
create a IngressClaim resource:

```
type IngressClaim struct {
    unversioned.TypeMeta `json:",inline"`
    ObjectMeta           `json:"metadata,omitempty"`
    
    Spec IngressClaimSpec
    Status IngressClaimStatus
}

type IngressClaimSpec struct {
    Ingresses   []LocalObjectReference //reference to Ingress Resources
    IngressReplicaSetSpec ReplicaSetSpec
}

type IngressClaimStatus struct {
    IngressReplicaSetName string
}
```

* No Ingress scheduling process will be envolved, Ingress Reources in IngressClaim
  are directly bound to the Ingress ReplicaSet auto provisioned.
* If all Ingress Resources in IngressClaim are deleted, IngressClaim will be 
  retained/recycled/deleted based on some policy specified by user
* Add a IngressClaimController in ControllerManager to sync IngressClaim resource,
  it works in the way similiar with PersistentVolumeClaimController: auto 
  provision Ingress ReplicaSet based on IngressClaim; retain/recycle/delete 
  IngressClaim if it's referenced Ingress Resources are deleted. 


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/job.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
